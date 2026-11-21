import copy
import unittest
from types import SimpleNamespace
from benchmark.core.cpu_threads import choose_winners, balanced, jobs_for_stage1, stats, applied_session

class CPUThreadsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(primary_models=['m'],modes=['batch'],repeats=5,intra_op_candidates=[1,2,3],baseline_intra_op=2)
        self.rows=[]
        for intra in [1,2,3]:
            for repeat in range(5):
                self.rows.append(dict(stage=1,model_id='m',mode='batch',receipt_id='r',intra_op_threads=intra,
                    status='OK',ocr_loop_ms={1:15,2:20,3:10}[intra],quality={'cell_metrics':[
                    dict(field='printed_total',type='numeric',correct=intra!=3,expected='1',observed='1' if intra!=3 else '2'),
                    dict(field='row:1:name',type='text',char_edits=0,expected='Товар Brand',observed='Товар Brand')]}))
    def test_accuracy_regression_disqualifies_fastest(self):
        self.assertEqual(choose_winners(self.rows,self.cfg),{'m/batch':1})
    def test_unstable_or_missing_repeats_disqualify(self):
        self.rows[0]['status']='UNSTABLE_BACKGROUND_CPU'
        self.assertEqual(choose_winners(self.rows,self.cfg),{'m/batch':2})
        self.rows.pop(5)
        self.assertEqual(choose_winners(self.rows,self.cfg),{})
    def test_balance_and_stats(self):
        self.assertEqual(balanced([1,2,3,4,6,8]),[1,8,2,6,3,4])
        self.assertEqual(stats([1,5,3,4,2]),dict(median=3,p95=5,min=1,max=5,spread=4,n=5))
    def test_not_applied(self):
        import cv2
        opts=SimpleNamespace(intra_op_num_threads=2,inter_op_num_threads=1,execution_mode='sequential')
        session=SimpleNamespace(get_session_options=lambda:opts,get_providers=lambda:['CPUExecutionProvider'])
        engine=SimpleNamespace(_load_rec_model=lambda:SimpleNamespace(session=SimpleNamespace(session=session)))
        actual,valid=applied_session(engine,3,1,cv2.getNumThreads())
        self.assertFalse(valid)
        self.assertEqual(actual['intra_op_threads'],2)

    def test_report_keeps_thread_configurations_separate(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from benchmark.core.cpu_threads import report
        from benchmark.core.runner import ROOT,load_json
        cfg=load_json(ROOT/'config/cpu_threads.json')
        cfg.update(primary_models=['m'],secondary_models=[],modes=['batch'])
        rows=copy.deepcopy(self.rows)
        for row in rows:
            row.update(opencv_threads=2,wall_ms=row['ocr_loop_ms']+5,process_cpu_ms=12,
                       cpu_utilization_percent=150,peak_ram_mb=100,numeric_accuracy=1,text_cer=0)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'config_snapshot.json').write_text(json.dumps(cfg),encoding='utf-8')
            (p/'job_000.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            with patch('benchmark.core.cpu_threads.OUTPUT',p):report()
            data=json.loads((p/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data['aggregates']),3)
            self.assertTrue(all(a['ocr_loop_ms']['n']==5 for a in data['aggregates']))
            self.assertIn('<svg',(p/'report.html').read_text(encoding='utf-8'))

    def test_retry_replaces_only_matching_unstable_run(self):
        from benchmark.core.cpu_threads import active_rows
        original=dict(self.rows[0],run_number=1,status='UNSTABLE_BACKGROUND_CPU',opencv_threads=2)
        replacement=dict(original,status='OK',is_retry=True)
        other=dict(original,run_number=2,status='OK')
        raw=[original,other,replacement]
        selected=active_rows(raw)
        self.assertEqual(len(raw),3)
        self.assertEqual(len(selected),2)
        self.assertTrue(all(r['status']=='OK' for r in selected))
    def test_schedule_alternates_individual_session_thread_counts(self):
        cfg=dict(intra_op_candidates=[1,2,3,4,6,8],primary_models=['a','b'],opencv_threads=2,modes=['single_cell','batch'])
        jobs=jobs_for_stage1(cfg)
        self.assertEqual([j['intra'] for j in jobs],[1,8,2,6,3,4]*2)
        self.assertEqual(len({(j['model_id'],j['intra']) for j in jobs}),12)
