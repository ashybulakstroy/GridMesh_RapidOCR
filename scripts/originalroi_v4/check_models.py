#!/usr/bin/env python3
"""Read configured RapidOCR models without downloading or running them."""
from __future__ import annotations
import argparse,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    project=p.parse_args().project.resolve()
    sys.path.insert(0,str(project))
    from App.ocr.experiment_models import ExperimentModelPool,inspect_model
    pool=ExperimentModelPool(root=project)
    print('Enabled:',pool.stats()['enabled'])
    for task in ('numeric','footer','vat'):
        print(task+':',', '.join(pool.models_for(task)) or '(disabled)')
        for mid in pool.models_for(task):
            print(' ',mid,inspect_model(pool.catalog[mid],project))
    print('This check never downloads or starts OCR models.')

if __name__=='__main__':main()
