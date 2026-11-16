"""Summarize AE reconstruction L1 for the private, preselected 24 validation groups."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generator',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True,help='Private frozen JSON list of exactly 24 validation sample IDs')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    wanted=json.loads(a.selection.read_text())
    if len(wanted)!=24 or len(set(wanted))!=24:
        raise ValueError('Expected 24 unique, preselected validation IDs')
    summary=json.loads((a.generator/'ae/training_summary.json').read_text())
    if not summary['training_complete'] or summary['test_split_used'] is not False:
        raise ValueError('Expected completed TRAIN/VAL autoencoder')
    epoch=summary['selection']['best_epoch']
    seen={}
    for shard in (a.generator/('ae/validation/epoch_%03d'%epoch)).glob('rank*.jsonl'):
        with shard.open() as stream:
            for line in stream:
                row=json.loads(line)
                key=row['sample_id']
                if key in wanted:
                    if key in seen or not math.isfinite(row['l1']):
                        raise ValueError('Duplicate or invalid validation measurement')
                    seen[key]=row['l1']
    if len(seen)!=24:
        raise ValueError('Preselected validation group missing')
    with a.output.open('x',encoding='utf-8') as out:
        json.dump(dict(selected_epoch=epoch,n=24,mean_l1=statistics.fmean(seen[i] for i in wanted)),out,indent=2)


if __name__=='__main__':
    main()
