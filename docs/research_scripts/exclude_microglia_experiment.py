"""Prepare and summarize the six-class rerun excluding one microglial cell."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = '864691135587086972'
SOURCE = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_sixclass_geo10um_v1')
WORK = SOURCE.with_name(SOURCE.name.replace('_v1', f'_exclude_{ROOT}_v1'))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare():
    if (WORK / 'data_verification.json').exists():
        raise RuntimeError('Prepared experiment already exists; inspect before rerunning')
    verified = read(SOURCE / 'data_verification.json')
    assert verified['complete'] and read(SOURCE / 'complete.json')['complete']
    for path, digest in verified['code_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    folds = read(SOURCE / 'folds.json')
    excluded_fold = folds['assignments'].pop(ROOT)
    excluded_counts = np.zeros(6, dtype=np.int64)
    found = 0
    cell_ids = set()
    for folder in ['soma_cells', 'geodesic/cells']:
        destination = WORK / 'data' / folder
        destination.mkdir(parents=True, exist_ok=True)
        for src in sorted((SOURCE / 'data' / folder).glob('*.npz')):
            if src.stem == ROOT:
                with np.load(src) as z:
                    assert str(z['root_id']) == ROOT
                    excluded_counts += np.bincount(z['y'], minlength=6)
                found += 1
                continue
            dst = destination / src.name
            if not dst.exists():
                dst.symlink_to(src.resolve())
            assert dst.is_symlink() and dst.resolve() == src.resolve()
            cell_ids.add(src.stem)
    assert found == 1 and excluded_counts.tolist() == [0, 0, 0, 0, 863, 0]
    assert cell_ids == set(folds['assignments']) and ROOT not in cell_ids
    for row in folds['summaries']:
        subset = 'test' if row['fold'] == excluded_fold else 'train'
        row[f'n_{subset}_cells'] -= 1
        row[f'{subset}_counts'] = (np.asarray(row[f'{subset}_counts']) - excluded_counts).tolist()
        assert min(row['train_counts']) > 0 and min(row['test_counts']) > 0
    folds['policy'] += f'; exclude {ROOT} entirely, retaining every other assignment'
    folds['source_six_class_fold_plan'] = str(SOURCE / 'folds.json')
    write(WORK / 'folds.json', folds)
    manifest = read(SOURCE / 'data/manifest.json')
    manifest['excluded_cell'].append(ROOT)
    manifest['added_cells'] = [r for r in manifest['added_cells'] if r['root_id'] != ROOT]
    write(WORK / 'data/manifest.json', manifest)
    verified['n_cells'] -= 1
    verified['n_points'] -= int(excluded_counts.sum())
    verified['class_counts'] = (np.asarray(verified['class_counts']) - excluded_counts).tolist()
    verified.update(source_six_class_run=str(SOURCE), excluded_root_id=ROOT,
                    excluded_class_counts=excluded_counts.tolist(), excluded_original_fold=excluded_fold,
                    preparation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    data_policy='Exact source array symlinks except excluded cell; no feature regeneration')
    write(WORK / 'data_verification.json', verified)
    print(json.dumps(verified, indent=2))


def summarize():
    os.environ['SUBCOMPARTMENT_SIX_WORK'] = str(WORK)
    import glia_subcompartment
    glia_subcompartment.summarize()
    previous = read(SOURCE / 'complete.json')
    current = read(WORK / 'complete.json')
    classes = current['classes']
    comparison = dict(excluded_root_id=ROOT, source_run=str(SOURCE), work=str(WORK),
                      previous_mean_macro_f1=previous['mean_macro_f1'],
                      rerun_mean_macro_f1=current['mean_macro_f1'], per_class={})
    for cls in classes:
        comparison['per_class'][cls] = {
            'previous_mean_f1': float(np.mean([f['per_class'][cls]['f1-score'] for f in previous['folds']])),
            'rerun_mean_f1': float(np.mean([f['per_class'][cls]['f1-score'] for f in current['folds']]))}
    comparison['evaluation'] = ('Cell-disjoint folds also select best epoch. The rerun changes both training '
                                'and evaluation cohorts; differences are not a pure training-effect estimate.')
    current['excluded_root_id'] = ROOT
    current['source_six_class_run'] = str(SOURCE)
    write(WORK / 'complete.json', current)
    write(WORK / 'comparison_previous.json', comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'summarize'])
    args = parser.parse_args()
    {'prepare': prepare, 'summarize': summarize}[args.stage]()
