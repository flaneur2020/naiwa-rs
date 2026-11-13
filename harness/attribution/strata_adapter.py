#!/usr/bin/env python3
"""Materialize an owned per-arm config then exec the actual Strata server."""
import json
import os
from pathlib import Path
import sys

out=Path(sys.argv[1]).resolve()
config=json.loads(Path(__file__).with_name('strata-nospec-template.json').read_text())
if len(sys.argv)>2:
    suffix=int(sys.argv[2])
    assert suffix>=0
    config['args'][config['args'].index('--suffix-draft')+1]=str(suffix)
config['log']=str(out/'strata-engine.log')
if os.environ.get('STRATA_WORK_LEDGER_FILE'):
    config['exe']='/home/yazhou/code/strata/build/strata'
    config['env']={**config.get('env',{}), 'STRATA_WORK_LEDGER_FILE':str(out/'strata-ledger.jsonl'),
                   'STRATA_VERIFY_PROFILE':'1', 'STRATA_DECODE_TIMING':'1'}
path=out/'strata-config.json'
path.write_text(json.dumps(config,indent=2))
os.execv(sys.executable,[sys.executable,'serve/server.py','--engine','strata','--config',str(path),'--port',str(config['port'])])
