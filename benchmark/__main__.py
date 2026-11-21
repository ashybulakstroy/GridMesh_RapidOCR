"""Run as python -m benchmark from repo root."""
import argparse,sys
from .core.runner import plan,run,check_inputs,read_ground_truth
from .core.report import write_report
from .core.runner import ROOT

def main():
    parser=argparse.ArgumentParser(description='Independent locally-measured RapidOCR benchmark')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('validate',help='check GT math, input files, ROI and hash; no OCR')
    sub.add_parser('list',help='list known GT records')
    p=sub.add_parser('plan',help='show workload without running models')
    for t in (p,):
        t.add_argument('--scope',choices=['quick','all'],default='quick')
        t.add_argument('--mode',choices=['both','single_cell','batch'],default='both')
        t.add_argument('--sweep',choices=['none','threads','batch'],default='none')
        t.add_argument('--repeats',type=int,default=None,help='measured repeats; default from config')
    p=sub.add_parser('run',help='perform real local measurements; requires explicit approval')
    p.add_argument('--crop-source',choices=['roi','gridmesh_total'],default='roi')
    p.add_argument('--confirm-load',action='store_true')
    p.add_argument('--confirm-sweep',action='store_true')
    p.add_argument('--scope',choices=['quick','all'],default='quick')
    p.add_argument('--mode',choices=['both','single_cell','batch'],default='both')
    p.add_argument('--sweep',choices=['none','threads','batch'],default='none')
    p.add_argument('--repeats',type=int,default=None,help='measured repeats; default from config')
    p=sub.add_parser('report',help='render already saved measurements; no OCR')
    p.add_argument('--run-id',required=True)
    args=parser.parse_args()
    try:
        if args.command=='validate':
            _,problems=check_inputs();return 2 if problems else 0
        if args.command=='list':
            for gt in read_ground_truth().values():print(f"{gt['receipt_id']}: {gt['source_image']} {len(gt['products'])} rows")
            return 0
        if args.command=='plan':plan(args.scope,args.mode,args.sweep,args.repeats);return 0
        if args.command=='run':
            run(scope=args.scope,mode=args.mode,sweep=args.sweep,repeats=args.repeats,confirm_load=args.confirm_load,confirm_sweep=args.confirm_sweep,crop_source=args.crop_source);return 0
        if args.command=='report':
            root=(ROOT/'data'/'output').resolve();directory=(root/args.run_id).resolve()
            if not directory.is_relative_to(root) or not directory.is_dir():raise ValueError('unknown run_id')
            r=write_report(directory);print('LOCAL REPORT:',directory/'report.html',r['successful_trials']);return 0
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
