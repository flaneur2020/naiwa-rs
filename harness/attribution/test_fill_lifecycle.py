import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fill_lifecycle import analyze

class LifecycleTests(unittest.TestCase):
    def test_superseded_used_and_censored(self):
        records=[]
        def event(phase,expert,ns,bytes=0):
            records.append(dict(layer=0,slot=1,tick=3,phase=phase,expert=expert,ns=ns,bytes=bytes))
        event('cache_fill_h2d',2,1,100);event('fill_publish',2,2)
        event('cache_fill_h2d',3,3,100);event('fill_publish',3,4)
        event('resident_use',3,5);event('promotion_publish',4,6)
        event('cache_fill_h2d',5,7,100);event('fill_publish',5,8)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events';p.write_text('\n'.join(map(json.dumps,records)))
            with patch('fill_lifecycle.schedule'):r=analyze(p)
        self.assertAlmostEqual(r['unused_superseded_byte_fraction'],1/3)
        self.assertEqual(r['groups']['right_censored_unused']['count'],1)
        self.assertEqual(r['groups']['superseded_by_publication_used']['count'],1)
        self.assertEqual(r['unpaired_publications'],0)
