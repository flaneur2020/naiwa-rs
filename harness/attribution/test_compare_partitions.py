import json
from pathlib import Path
import tempfile
import unittest
from compare_partitions import naiwa,PHASES

class PartitionTests(unittest.TestCase):
    def test_complete_groups_and_drop(self):
        records=[dict(layer=l,rows=1,endpoints_ms=[[p,i*.01] for i,p in enumerate(PHASES)]) for l in range(48)]
        records.append(records[0])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events'
            p.write_text('\n'.join(map(json.dumps,records)))
            r=naiwa(p)
            self.assertEqual(r['complete_groups'],1)
            self.assertAlmostEqual(r['span_ms'],48*.09)
            records.append(dict(flush=True,dropped=1));p.write_text('\n'.join(map(json.dumps,records)))
            with self.assertRaisesRegex(ValueError,'dropped'):naiwa(p)
