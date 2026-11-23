"""No network, OCR inference, or fake performance records in these checks."""
import json, unittest
from pathlib import Path
from benchmark2.runner import metric, levenshtein
from benchmark2.prepare import ROOT
from benchmark2.report import build_report

class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.samples=json.loads((ROOT/'data/samples.json').read_text(encoding='utf-8'))
        self.models=json.loads((ROOT/'config/models.json').read_text(encoding='utf-8'))
        self.config=json.loads((ROOT/'config/benchmark.json').read_text(encoding='utf-8'))
    def test_six_real_cyrillic_engines(self):
        self.assertEqual(len(self.models),6)
        self.assertTrue(all(m['cyrillic'] for m in self.models))
        self.assertEqual({m['kind'] for m in self.models},{'rapid','tesseract','windows'})
        self.assertIn('v6_medium_cyrillic',{m['id'] for m in self.models})
        self.assertNotIn('v6_small_official',{m['id'] for m in self.models})
    def test_three_passes(self):
        self.assertEqual(self.config['total_passes'],3)
        self.assertEqual(self.config['warmup_passes'],1)
        self.assertEqual(self.config['measured_passes'],2)
    def test_17_source_cells(self):
        self.assertEqual(len(self.samples),17)
        self.assertEqual(len({s['id'] for s in self.samples}),17)
        self.assertEqual(sum(s['type']=='name' for s in self.samples),11)
        self.assertEqual(sum(s['type']=='label' for s in self.samples),3)
        self.assertEqual(sum(s['type']=='numeric' for s in self.samples),3)
        self.assertEqual(len({s['source'] for s in self.samples}),3)
        self.assertTrue(all(len(s['image_sha256'])==64 for s in self.samples))
        self.assertTrue(all(len(s['bbox'])==4 and s['expected'] for s in self.samples))
    def test_cyrillic_metrics_not_transliterated(self):
        self.assertEqual(metric('Перчатки 1000','Перчатки 1000')['edit_distance'],0)
        self.assertGreater(metric('Кабель','Kaбeль')['edit_distance'],0)
        self.assertEqual(levenshtein('тест','тест'),0)
    def test_report_no_fabricated_timings(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)
            build_report(f,self.samples,self.models,[],[],{'run_id':'TEST_NO_INFERENCE'},complete=False)
            text=(f/'report.html').read_text(encoding='utf-8')
            self.assertIn('17 ячеек',text)
            self.assertIn('Нет измеренных результатов',text)
            self.assertIn('PP-OCRv6 Medium Cyrillic',text)
            self.assertEqual(len(json.loads((f/'summary.json').read_text(encoding='utf-8'))),6)
if __name__=='__main__':unittest.main()

class TestPipelineSmoke(unittest.TestCase):
    """Synthetic data used ONLY to validate control flow; never counted as OCR benchmark."""
    def test_run_report_and_306_records_with_stub(self):
        import tempfile, shutil
        from unittest.mock import patch
        from PIL import Image
        import benchmark2.runner as b
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'config').mkdir();(root/'data/cells').mkdir(parents=True)
            (root/'results').mkdir()
            for p in ('samples.json',):shutil.copy2(ROOT/'data'/p,root/'data'/p)
            for p in ('models.json','benchmark.json'):shutil.copy2(ROOT/'config'/p,root/'config'/p)
            samples=json.loads((root/'data/samples.json').read_text(encoding='utf-8'))
            manifest=[]
            for s in samples:
                # Test-only fixture, NOT a receipt, NOT shipped as benchmark input.
                f=root/'data/cells'/(s['id']+'.png')
                Image.new('RGB',(80,22),'white').save(f)
                from benchmark2.prepare import sha
                manifest.append(dict(s,file='data/cells/'+f.name,width=80,height=22,
                                     crop_sha256=sha(f)))
            (root/'data/cells_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
            class FakeAdapter:
                def recognize(self, im): return 'тест', .99
                def close(self):pass
            with patch.object(b,'ROOT',root),patch.object(b,'make_adapter',side_effect=lambda *a:FakeAdapter()),patch.object(b,'precompute',side_effect=lambda m,r:{s['id']:[Image.new('RGB',(80,22),'white')] for s in m}):
                dest=b.run(auto_prepare=False)
            entries=(dest/'raw.jsonl').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(entries),306)
            self.assertTrue((dest/'report.html').is_file())
            self.assertTrue((dest/'raw.csv').is_file())
            self.assertEqual(json.loads((dest/'status.json').read_text(encoding='utf-8'))['percent'],100)
            rows=[json.loads(x) for x in entries]
            self.assertEqual(len([r for r in rows if r['phase']=='warmup']),102)
            self.assertEqual(len([r for r in rows if r['phase']=='measured']),204)
