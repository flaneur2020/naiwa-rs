"""Shared, stdlib-only measurement primitives. Times are never inferred from chunk counts."""
import hashlib
import json
import statistics
import time
import urllib.request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def capture(url, body, timeout=1200):
    body = dict(body, stream=True, stream_options={'include_usage': True})
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter_ns()
    events, usage, finish, done = [], None, [], False
    logprobs=[]
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            stamp = time.perf_counter_ns()
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise ValueError(f'server stream error: {event["error"]}')
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('finish_reason'):
                    finish.append(choice['finish_reason'])
                if choice.get('logprobs') is not None:
                    logprobs.append(choice['logprobs'])
                text = choice.get('delta', {}).get('content')
                if text:
                    events.append({'offset_ns': stamp-start, 'text': text})
    end = time.perf_counter_ns()
    if not done or not events:
        raise ValueError('incomplete stream or no content')
    ct = (usage or {}).get('completion_tokens')
    if not isinstance(ct, int) or ct <= 1:
        raise ValueError('server completion_tokens missing/invalid; chunk fallback forbidden')
    text = ''.join(e['text'] for e in events)
    first = events[0]['offset_ns']
    return dict(schema=1, text=text, sha256=hashlib.sha256(text.encode()).hexdigest(),
                events=events, usage=usage, finish_reasons=finish, logprobs=logprobs,
                content_chunks=len(events), decode_intervals=ct-1,
                ttft_ms=first/1e6, total_ms=(end-start)/1e6,
                tpot_ms=(end-start-first)/1e6/(ct-1),
                timing_boundary='first nonempty content -> DONE received; includes SSE transport',
                sse_content_gaps_ms=[(b['offset_ns']-a['offset_ns'])/1e6
                                     for a,b in zip(events,events[1:])])


def summarize(records):
    """Per-process median is the sample unit, not pooled requests."""
    values = [r['tpot_ms'] for r in records]
    return {'median_tpot_ms': statistics.median(values), 'request_count': len(records),
            'sha256': [r['sha256'] for r in records],
            'within_process_text_equal': len({r['sha256'] for r in records}) == 1}
