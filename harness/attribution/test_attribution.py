import contextlib
import http.server
import io
import json
from pathlib import Path
import tempfile
import socket
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from analyze import compare_routes, cpu, decompose_count_cost, gpu, report, schedule, union
from common import capture
from run import validate


class Tests(unittest.TestCase):
    def test_cpu_uses_window_delta_not_cumulative_count(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cpu.log'
            p.write_text('ROUND_STATS 4 50 0 50 0 120\nROUND_STATS 8 50 0 50 0 160\nCPU_TASK layer=0 tick=1 rows=1 submit_us=10 worker_us=20 compute_return_us=120 ids=[]\n')
            result=cpu(p)
            self.assertEqual(result['round_windows'][1]['cpu_job_wall_ms_per_pass'],40)
            self.assertEqual(result['worker_to_return_us']['p50'],100)

    def test_schedule_rejects_incomplete_or_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'schedule.jsonl'
            events=[dict(layer=0,tick=3,phase=k,ns=i*1000) for i,k in enumerate(['route_ready','cpu_submit','dma_enqueued','cpu_submit_return','cpu_worker','cpu_compute_return'])]
            p.write_text('\n'.join(json.dumps(e) for e in events+[{'flush':True,'dropped':0}]))
            self.assertEqual(schedule(p)['intervals']['route_to_submit']['mean_us'],1)
            p.write_text('\n'.join(json.dumps(e) for e in events+[{'flush':True,'dropped':1}]))
            with self.assertRaises(ValueError):schedule(p)

    def test_exact_count_cost_decomposition(self):
        d=decompose_count_cost(55,5000,58,5200,127)
        self.assertAlmostEqual(d['count_component']+d['cost_component'],d['total_delta'])
        self.assertGreater(d['count_component'],0)
        self.assertLess(d['cost_component'],0)

    def test_mtp_records_validation(self):
        from mtp import requests
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            record={'seq_id':7,'terminal':True,'steps':[[2,1,20],[2,2,23]]}
            (p/'mtp.jsonl').write_text(json.dumps(record)+'\n')
            (p/'request-1.json').write_text(json.dumps({'usage':{'completion_tokens':128},'total_ms':1000,'ttft_ms':100,'tpot_ms':900/127,'sha256':'x'*64}))
            out=requests(p,1)
            self.assertEqual(out[0]['steps'],2)
            self.assertEqual(out[0]['accepted'],3)
            self.assertEqual(out[0]['decode_intervals'],127)
            record['steps'][1][2]=19
            (p/'mtp.jsonl').write_text(json.dumps(record)+'\n')
            with self.assertRaisesRegex(ValueError,'keep_len'):requests(p,1)
            record['steps'][1]=[2,3,23]
            (p/'mtp.jsonl').write_text(json.dumps(record)+'\n')
            with self.assertRaisesRegex(ValueError,'invalid'):requests(p,1)

    def test_union(self):
        self.assertEqual(union([(0,10),(3,5),(9,20),(20,21),(22,23)]),[(0,21),(22,23)])

    def test_gpu_overlap_clipping_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.csv'
            p.write_text('preamble\nStart (ns),Duration (ns),Name\n0,100,A\n10,10,nested\n20,100,[CUDA memcpy Host-to-Device]\n150,50,B\n')
            r=gpu(p,0,200,1,1)
            self.assertAlmostEqual(r['busy_union_ms_per_step'],170/1e6)
            self.assertAlmostEqual(r['classes']['copy']['exclusive_ms_per_step'],20/1e6)
            self.assertEqual(r['gap_classes'][0]['before'],'[CUDA memcpy Host-to-Device]')
            self.assertEqual(r['gap_classes'][0]['after'],'B')
            clipped=gpu(p,50,160,1,1)
            self.assertAlmostEqual(clipped['busy_union_ms_per_step'],80/1e6)

    def test_route_difference_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            rec={'layer':0,'tick':1,'ids':[[2]],'weight_bits':[[123]]}
            a.write_text('other\nMOE_WORK '+json.dumps(rec)+'\n')
            b.write_text(a.read_text())
            self.assertTrue(compare_routes(a,b)['equal'])
            rec['weight_bits']=[[124]]
            b.write_text('MOE_WORK '+json.dumps(rec)+'\n')
            self.assertEqual(compare_routes(a,b)['first_difference']['index'],0)
            b.write_text('')
            self.assertFalse(compare_routes(a,b)['equal'])

    def test_stream_usage_denominator_not_chunks(self):
        payload=[b'data: {"choices":[{"delta":{"content":"abc"}}]}\n',
                 b'data: {"choices":[{"delta":{"content":"def"}}],"usage":{"completion_tokens":128}}\n',
                 b'data: [DONE]\n']
        with patch('urllib.request.urlopen',return_value=contextlib.nullcontext(payload)):
            r=capture('http://localhost',{})
        self.assertEqual(r['decode_intervals'],127)
        self.assertEqual(r['content_chunks'],2)
        self.assertEqual(len(r['sha256']),64)
        self.assertEqual(r['text'],'abcdef')
        with patch('urllib.request.urlopen',return_value=contextlib.nullcontext(payload[:1]+payload[-1:])):
            with self.assertRaisesRegex(ValueError,'completion_tokens'):
                capture('http://localhost',{})
        with patch('urllib.request.urlopen',return_value=contextlib.nullcontext(payload[:-1])):
            with self.assertRaisesRegex(ValueError,'incomplete'):
                capture('http://localhost',{})

    def test_runner_restarts_and_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            server=root/'server.py'
            server.write_text('''import http.server,json,sys
class H(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"fake"}]}')
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']))
  self.send_response(200);self.end_headers()
  n=128 if sys.argv[2]=='ok' else 7
  event={'choices':[{'delta':{'content':'hello'}}],'usage':{'completion_tokens':n}}
  self.wfile.write(('data: '+json.dumps(event)+'\\n\\ndata: [DONE]\\n\\n').encode())
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
''')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
            config=dict(rounds=3,requests=1,mode='clean',drafting='off',
                        body={'max_tokens':128,'temperature':0,'seed':0},
                        arms=[dict(name=n,cwd=d,url=f'http://127.0.0.1:{port}',
                                   argv=[sys.executable,str(server),str(port),'ok']) for n in ('a','b')])
            cp=root/'config.json';cp.write_text(json.dumps(config))
            runner=Path(__file__).with_name('run.py')
            done=subprocess.run([sys.executable,str(runner),str(cp),'--out',str(root/'ok')],capture_output=True,text=True,timeout=45)
            self.assertEqual(done.returncode,0,done.stderr)
            manifest=json.loads((root/'ok/manifest.json').read_text())
            self.assertEqual(len(manifest['samples']),6)
            self.assertEqual([s['arm'] for s in manifest['samples']],['a','b','b','a','a','b'])
            self.assertTrue(manifest['all_text_equal'])
            config['arms'][0]['argv'][-1]='short';cp.write_text(json.dumps(config))
            failed=subprocess.run([sys.executable,str(runner),str(cp),'--out',str(root/'fail')],capture_output=True,timeout=15)
            self.assertNotEqual(failed.returncode,0)
            manifest=json.loads((root/'fail/manifest.json').read_text())
            self.assertEqual(manifest['status'],'failed')
            self.assertEqual(len(manifest['samples']),0)
            self.assertEqual(len(list((root/'fail').glob('r*'))),1)

    def test_config_mtp_gate(self):
        c=dict(rounds=3,requests=1,mode='clean',body={'max_tokens':128,'temperature':0,'seed':0},drafting='on',arms=[{'name':'a','argv':[]},{'name':'b','argv':[]}])
        with self.assertRaisesRegex(AssertionError,'explicit MTP'):
            validate(c)

    def test_report_diagnostic_is_not_perf(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            p.write_text(json.dumps({'status':'complete','mode':'diagnostic','samples':[
                {'arm':arm,'round':r,'median_tpot_ms':t} for r in (1,2,3) for arm,t in [('a',10),('b',9)]]}))
            out=report(p)
            self.assertEqual(out['median_paired_delta_ms'],-1)
            self.assertFalse(out['performance_eligible'])


if __name__=='__main__':unittest.main()
