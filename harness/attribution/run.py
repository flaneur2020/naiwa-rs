#!/usr/bin/env python3
"""Restarted alternating A/B, owned process groups only, no retries after readiness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from common import capture, digest, summarize


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def validate(config):
    assert config['rounds'] >= 3, 'at least three restart rounds required'
    assert config['requests'] >= 1
    assert len(config['arms']) in (2,3)
    names = [arm['name'] for arm in config['arms']]
    assert len(set(names)) == len(names)
    assert all(name and all(c.isalnum() or c in '-_' for c in name) for name in names), 'safe unique arm names required'
    assert config['mode'] in ('clean', 'diagnostic')
    assert config['body']['max_tokens'] == 128, 'fixed 128-token experiment'
    assert config['body'].get('temperature') == 0
    assert 'seed' in config['body']
    assert config.get('drafting') in ('off', 'on')
    for arm in config['arms']:
        drafting = arm.get('drafting', config['drafting'])
        assert drafting in ('off','on')
        if drafting == 'on':
            assert '--mtp-n-predict' in arm['argv'], 'explicit MTP depth required'
        else:
            assert '--mtp' not in arm['argv'], 'drafting=off conflicts with --mtp'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    args.out.mkdir(parents=True, exist_ok=False)  # never overwrite/retry a failed run
    (args.out/'config.json').write_text(json.dumps(config, indent=2))
    manifest = dict(schema=1, status='running', mode=config['mode'], config_sha256=digest(config),
                    samples=[], no_retry=True, drafting=config['drafting'])
    def save():
        (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    save()
    try:
        for rnd in range(1, config['rounds']+1):
            if len(config['arms']) == 2:
                arms = config['arms'] if rnd % 2 else list(reversed(config['arms']))
            else:
                offset = (rnd-1) % len(config['arms'])
                arms = config['arms'][offset:] + config['arms'][:offset]
            for index, arm in enumerate(arms):
                tag = f'r{rnd}-{arm["name"]}'
                directory = args.out/tag
                directory.mkdir()
                env = os.environ.copy()
                # Hermetic experiment knobs: ambient NAIWA/MISTRALRS values are not inherited.
                for key in list(env):
                    if key.startswith(('NAIWA_', 'MISTRALRS_')):
                        del env[key]
                env.update(config.get('env', {}))
                env.update(arm.get('env', {}))
                if config['mode'] == 'clean':
                    allowed = set(config.get('clean_env_allowlist', []))
                    suspects = [k for k in env if k.startswith(('NAIWA_', 'MISTRALRS_'))
                                and any(s in k for s in ('DBG','TRACE','TIMELINE','STATS','STEP_TIME','ROUTE_IDS','XSUM','PROFILE','SCHEDULE','MTP_TRACE'))
                                and k not in allowed]
                    if suspects:
                        raise ValueError(f'diagnostic env in clean run: {suspects}')
                if env.get('NAIWA_WORK_TRACE') == '1':
                    env['NAIWA_WORK_TRACE_FILE'] = str((directory/'work.jsonl').resolve())
                if 'NAIWA_MTP_TRACE_FILE' in env:
                    env['NAIWA_MTP_TRACE_FILE'] = str((directory/'mtp.jsonl').resolve())
                if 'NAIWA_SCHEDULE_FILE' in env:
                    env['NAIWA_SCHEDULE_FILE'] = str((directory/'schedule.jsonl').resolve())
                if 'NAIWA_GPU_PARTITION_FILE' in env:
                    env['NAIWA_GPU_PARTITION_FILE'] = str((directory/'gpu-partition.jsonl').resolve())
                if 'NAIWA_MODEL_PARTITION_FILE' in env:
                    env['NAIWA_MODEL_PARTITION_FILE'] = str((directory/'model-partition.jsonl').resolve())
                if 'NAIWA_HEAD_PARTITION_FILE' in env:
                    env['NAIWA_HEAD_PARTITION_FILE'] = str((directory/'head-partition.jsonl').resolve())
                argv = [os.path.expanduser(x.replace('{run_dir}', str(directory.resolve()))) for x in arm['argv']]
                metadata = dict(round=rnd, arm=arm['name'], order=index, argv=argv,
                                cwd=os.path.expanduser(arm['cwd']), mode=config['mode'],
                                env={k:v for k,v in env.items() if k.startswith(('NAIWA_','MISTRALRS_','CUDA_'))})
                executable = Path(argv[0])
                if not executable.is_absolute():
                    executable = Path(metadata['cwd'])/executable
                if executable.is_file():
                    with executable.open('rb') as binary:
                        metadata['executable_sha256'] = hashlib.file_digest(binary, 'sha256').hexdigest()
                for key, command in [('git_head',['git','rev-parse','HEAD']),
                                     ('git_status',['git','status','--porcelain']),
                                     ('git_diff',['git','diff','--binary'])]:
                    result = subprocess.run(command,cwd=metadata['cwd'],capture_output=True)
                    if result.returncode == 0:
                        if key == 'git_diff':
                            metadata['git_diff_sha256'] = hashlib.sha256(result.stdout).hexdigest()
                        else:
                            metadata[key] = result.stdout.decode().strip()
                (directory/'launch.json').write_text(json.dumps(metadata, indent=2))
                origin = arm['url'].rstrip('/')
                try:
                    urllib.request.urlopen(origin+'/v1/models', timeout=2).close()
                except Exception:
                    pass
                else:
                    raise RuntimeError(f'{tag}: endpoint already serving; refusing attribution')
                with (directory/'engine.log').open('w') as log:
                    proc = subprocess.Popen(argv, cwd=metadata['cwd'], env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        deadline = time.monotonic()+config.get('startup_timeout_s',900)
                        while True:
                            if proc.poll() is not None:
                                raise RuntimeError(f'{tag}: engine exited {proc.returncode}')
                            try:
                                with urllib.request.urlopen(origin+'/v1/models', timeout=2) as response:
                                    model = json.load(response)['data'][0]['id']
                                break
                            except Exception:
                                if time.monotonic() > deadline:
                                    raise TimeoutError(f'{tag}: readiness timeout')
                                time.sleep(1)
                        records=[]
                        conversation=list(config['body'].get('messages',[]))
                        followups=config.get('chat_followups')
                        if followups is not None and len(followups)!=config['requests']-1:
                            raise ValueError('chat followups must match subsequent requests')
                        for i in range(config['requests']):
                            body=dict(config['body'], model=model)
                            if followups is not None:
                                if i: conversation.append({'role':'user','content':followups[i-1]})
                                body['messages']=list(conversation)
                            (directory/f'request-{i+1}-body.json').write_text(json.dumps(body,ensure_ascii=False,indent=2))
                            record=capture(origin+'/v1/chat/completions', body)
                            (directory/f'request-{i+1}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
                            if record['usage']['completion_tokens'] != 128:
                                raise ValueError(f'{tag}: short request; no retry')
                            records.append(record)
                            if followups is not None:
                                conversation.append({'role':'assistant','content':record['text']})
                        manifest['samples'].append(dict(metadata, **summarize(records)))
                        save()
                        if 'NAIWA_HEAD_PARTITION_FILE' in env:
                            time.sleep(2)
                            from head_costs import analyze as validate_head
                            validate_head(directory/'head-partition.jsonl')
                        if 'NAIWA_MTP_TRACE_FILE' in env:
                            from mtp import requests as validate_mtp_requests
                            validate_mtp_requests(directory, config['requests'])
                        if any(k in env for k in ('NAIWA_MODEL_PARTITION_FILE','NAIWA_GPU_PARTITION_FILE')):
                            # Allow idle background writers to flush before artifact validation.
                            time.sleep(2)
                        if 'NAIWA_MODEL_PARTITION_FILE' in env:
                            from model_costs import naiwa as validate_model_partition
                            if int(env.get('NAIWA_MODEL_PARTITION_STRIDE','1'))>1:
                                from model_costs import sparse
                                sparse(directory/'model-partition.jsonl')
                            else:
                                validate_model_partition(directory/'model-partition.jsonl')
                        if 'NAIWA_GPU_PARTITION_FILE' in env:
                            from compare_partitions import naiwa as validate_gpu_partition
                            validate_gpu_partition(directory/'gpu-partition.jsonl')
                        if 'STRATA_WORK_LEDGER_FILE' in env:
                            from ledger import strata as validate_strata_ledger
                            validate_strata_ledger(directory/'strata-ledger.jsonl')
                        if 'NAIWA_SCHEDULE_FILE' in env:
                            # Writer flushes only after 300ms without events. Do not truncate
                            # the last batch when terminating an otherwise successful arm.
                            time.sleep(2)
                            schedule_path = directory/'schedule.jsonl'
                            if not schedule_path.exists() or schedule_path.stat().st_size == 0:
                                raise RuntimeError('missing schedule artifact: instrumented build required')
                            from analyze import schedule as validate_schedule
                            validate_schedule(schedule_path)
                    finally:
                        stop(proc)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        save()
        raise
    manifest['status']='complete'
    for arm in config['arms']:
        hashes=[h for sample in manifest['samples'] if sample['arm']==arm['name'] for h in sample['sha256']]
        manifest.setdefault('text_stability',{})[arm['name']]=len(set(hashes))==1
    manifest['all_text_equal']=len({h for s in manifest['samples'] for h in s['sha256']})==1
    manifest['text_equal_by_request_position'] = {
        str(i+1): len({s['sha256'][i] for s in manifest['samples']}) == 1
        for i in range(config['requests'])
    }
    manifest['same_work_proven']=False
    manifest['warning']='Text equality is necessary but insufficient for identical expert/cache work. Pair diagnostic traces separately. SSE gaps are not decode-step timings.'
    save()
    print(json.dumps({k:v for k,v in manifest.items() if k!='samples'},indent=2))


if __name__ == '__main__':
    main()
