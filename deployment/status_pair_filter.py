"""Local progress for the historical same-subcompartment pair comparison."""
import argparse
import json
from pathlib import Path


def status(work):
    work=Path(work);plan=json.loads((work/'plan.json').read_text())
    done={p.stem for p in (work/'cells').glob('*.json')}
    errors={p.stem for p in (work/'errors').glob('*.json')}-done
    result=dict(work=str(work),neurons=plan['cells'],completed_neurons=len(done),
        excluded_neurons=plan['excluded_neurons'],label_ranks_complete=len(list((work/'label_status').glob('*.json'))),
        total_label_ranks=64,unresolved_errors=len(errors),error_cells=sorted(errors),complete=False)
    if (work/'summary.json').exists():result['complete']=json.loads((work/'summary.json').read_text())['complete']
    if (work/'jobs.json').exists():result['jobs']={k:v['job_id'] for k,v in json.loads((work/'jobs.json').read_text()).items()}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',default='/orcd/scratch/orcd/013/jcbliao/segclr_pair_audit/v117_subcompartment_filter_20261108')
    print(json.dumps(status(p.parse_args().work),indent=2))
