import unittest
from step_costs import window

class StepPartitionTests(unittest.TestCase):
    def sample(self):
        s=[0]*(48*33+4)
        for l in range(48):
            for i in [0,17,18,19,20,21,22,23,24]:s[l*33+i]=1000+l*100+i
        s[48*33]=5900;s[48*33+1]=6000
        return dict(pos=23,stamps=s,tail_sync_ms=1)
    def test_closure(self):
        r=window(self.sample())
        self.assertAlmostEqual(sum(r['segments_ms'].values()),r['gpu_stream_span_ms'])
    def test_missing_stamp_rejected(self):
        w=self.sample();w['stamps'][22]=0
        with self.assertRaises(ValueError):window(w)
