"""Read local completion receipts; do not scan the database prediction tables."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def status(work):
    work=Path(work)
    plan=json.loads((work/'plan.json').read_text())
    result=dict(work=str(work),family=plan['family'],roots=len(plan['roots']),target_nodes=plan['expected'],folds=len(plan['models']),complete=(work/'complete.json').exists())
    result['inference_complete']=(work/'inference_complete.json').exists()
    if (work/'stage_progress.json').exists():result['staging']=json.loads((work/'stage_progress.json').read_text())
    if (work/'inputs_complete.json').exists():result['staging']=json.loads((work/'inputs_complete.json').read_text())
    scored=defaultdict(int)
    for path in (work/'scores').glob('*/*/*.json'):
        row=json.loads(path.read_text())
        if row.get('complete'):scored[path.parent.parent.name]+=row['rows']
    result['scored_nodes_per_fold']=dict(scored)
    imports=defaultdict(lambda:defaultdict(int))
    for path in (work/'imports').glob('*/*/*.json'):
        row=json.loads(path.read_text())
        if row.get('complete'):imports[path.parent.parent.name][path.parent.name]+=row['rows']
    result['imported_nodes_by_fold']=dict(imports)
    if (work/'jobs.json').exists():
        ledger=json.loads((work/'jobs.json').read_text())
        result['jobs']={k:v['job_id'] for k,v in ledger['jobs'].items()}
        for key,retry in ledger.items():
            if key.startswith('retry_'):result['jobs'].update({k:v['job_id'] for k,v in retry['jobs'].items()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',default='/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/historical117_subcompartment_six_class_v1')
    print(json.dumps(status(p.parse_args().work),indent=2))
