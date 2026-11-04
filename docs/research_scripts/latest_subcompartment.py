"""Refresh the existing geodesic five-fold experiment with pinned SegCLR features."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
GNN = HERE.parent / 'gnn_classifier'
WORK = Path(os.environ.get('SUBCOMPARTMENT_LATEST_WORK', '/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/subcompartment_latest100k_geo10um_v1'))
SOURCE = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr/subcompartment_linear_v1')
FOLDS = GNN / 'results/subcompartment_geodesic_10um_cv/folds.json'
ARCHIVE = WORK.parent / 'latest100k_morpho6_centers_v1/vector_archive'
os.environ['LATEST_SEGCLR_WORK'] = str(WORK)
sys.path.insert(0, str(GNN))
from scripts import latest_segclr_features as f


def source_files():
    return sorted((SOURCE / 'soma_cells').glob('*.npz')) + sorted((SOURCE / 'geodesic/cells').glob('*.npz'))


def code_hashes():
    return {str(p): f.sha(p) for p in [Path(__file__), HERE / 'train_compartment_gpu.py', HERE / 'train_compartment_linear.py',
            HERE / 'prepare_compartment_training.py', GNN / 'gnn/resnet.py', Path(f.__file__), FOLDS]}


def check_plan():
    p = json.loads((WORK / 'plan.json').read_text())
    if p['code_hashes'] != code_hashes() or p['checkpoint_sha256'] != f.sha(f.WEIGHTS):
        raise ValueError('Pinned code, fold assignments, or embedding weights changed')
    return p


def plan():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from segclr_db import store as st
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / 'plan.json').exists():
        print('Existing verified plan', check_plan()['points'], flush=True)
        return
    if f.DB.exists():
        raise ValueError('Incomplete planning database requires inspection before retry')
    store = st.open_store('/orcd/compute/sdorkenw/001/segclr-db', 'microns')
    registry = st.scan(store, 'checkpoints', filter=f"run_id='{f.RUN}' AND checkpoint_id='{f.CHECKPOINT}'").to_pylist()
    if len(registry) != 1 or Path(registry[0]['path']).resolve() != f.WEIGHTS.resolve():
        raise ValueError('Checkpoint registry mismatch')
    inventory = []
    folds = json.loads(FOLDS.read_text())
    with f.open_db() as c:
        c.execute('CREATE TABLE requests(root_id BIGINT,x FLOAT,y FLOAT,z FLOAT)')
        for i, path in enumerate(source_files()):
            with np.load(path) as z:
                root = str(z['root_id'])
                if root not in folds['assignments']:
                    raise ValueError('Source cell missing from original folds')
                if (str(z['split']) != 'train') != (folds['assignments'][root] == 0):
                    raise ValueError('Original split drift')
                f.append_points(c, np.repeat(int(root), len(z['y'])), z['xyz_nm'])
                inventory.append(dict(path=str(path.relative_to(SOURCE)), sha256=f.sha(path), root_id=root,
                                      rows=len(z['y']), counts=np.bincount(z['y'], minlength=4).tolist()))
            if i % 250 == 0:
                print('Source samples', i, '/', len(source_files()), flush=True)
        c.execute('CREATE TABLE points AS SELECT row_number() OVER(ORDER BY root_id,x,y,z)::BIGINT AS point_id,* FROM (SELECT DISTINCT * FROM requests)')
        c.execute('DROP TABLE requests')
        points = c.execute('SELECT count(*) FROM points').fetchone()[0]
        # The current shared nodes table has unreadable fragments. Use the
        # already verified coordinate-bearing archive of this exact checkpoint.
        archived = json.loads((ARCHIVE/'complete.json').read_text())
        if archived['run_id'] != f.RUN or archived['checkpoint_sha256'] != f.sha(f.WEIGHTS):
            raise ValueError('Archived embedding provenance mismatch')
        archive_inventory = {str(path):f.sha(path) for path in sorted(ARCHIVE.glob('*.parquet'))}
        if not archive_inventory:
            raise ValueError('Verified feature archive is empty')
        c.read_parquet(list(archive_inventory)).create_view('shared_vectors')
        if c.execute('SELECT count(*) FROM shared_vectors').fetchone()[0] != archived['points']:
            raise ValueError('Archive coverage differs from verification receipt')
        print('Match exact cell/coordinates to verified newest feature archive', points, flush=True)
        c.execute('CREATE TABLE cached_raw AS SELECT p.point_id,e.embedding FROM shared_vectors e JOIN points p USING(root_id,x,y,z) WHERE e.embedding IS NOT NULL AND len(e.embedding)=64 AND isfinite(list_sum(e.embedding))')
        conflicts = c.execute('SELECT count(*) FROM (SELECT point_id FROM cached_raw GROUP BY point_id HAVING count(DISTINCT embedding)>1)').fetchone()[0]
        c.execute('CREATE TABLE cached AS SELECT point_id,first(embedding) AS embedding FROM cached_raw GROUP BY point_id HAVING count(DISTINCT embedding)=1')
        c.execute('DROP TABLE cached_raw')
        c.execute('CREATE TABLE missing AS SELECT p.* FROM points p ANTI JOIN cached c USING(point_id)')
        cached = c.execute('SELECT count(*) FROM cached').fetchone()[0]
        missing = c.execute('SELECT count(*) FROM missing').fetchone()[0]
        (WORK / 'requests').mkdir(exist_ok=True)
        chunks = []
        for i, batch in enumerate(c.execute('SELECT * FROM missing ORDER BY z,y,x,root_id').fetch_record_batch(8192)):
            path = WORK / 'requests' / f'{i:06d}.parquet'
            pq.write_table(pa.Table.from_batches([batch]), path, compression='zstd')
            chunks.append(dict(path=str(path), rows=len(batch), sha256=f.sha(path)))
        c.execute('CHECKPOINT')
    p = dict(experiment_id=f.EXPERIMENT, run_id=f.RUN, checkpoint_id=f.CHECKPOINT,
             checkpoint_path=str(f.WEIGHTS), checkpoint_sha256=f.sha(f.WEIGHTS), embedding_dim=64,
             points=points, cached_points=cached, missing_points=missing, ambiguous_cached_points_to_regenerate=conflicts,
             chunks=chunks, feature_archive=str(ARCHIVE), archive_inventory=archive_inventory,
             archive_receipt=archived,
             source=str(SOURCE), source_inventory=inventory, code_hashes=code_hashes(),
             sample_rows=sum(r['rows'] for r in inventory), n_folds=5,
             precision='tf32', em_volume='em_clahe2d_flash', seg_volume='seg_1718_zarr3_flash',
             change='Replace embeddings only; preserve every label, point ID, coordinate, and fold assignment')
    f.save(WORK / 'folds.json', folds)
    f.save(WORK / 'plan.json', p)
    print('PLAN READY', {k:p[k] for k in ['points','sample_rows','cached_points','missing_points']}, flush=True)


def assemble():
    import numpy as np
    p = check_plan()
    if not json.loads((WORK / 'preprocessing_parity.json').read_text())['passed']:
        raise ValueError('Preprocessing parity missing')
    output = WORK / 'data'
    audit = []
    with f.open_db() as c:
        f.create_feature_view(c)
        # One sorted archive permits row-group pruning for every cell lookup.
        archive = WORK / 'vectors.parquet'
        if not archive.exists():
            temp = archive.with_suffix('.tmp.parquet')
            c.execute('COPY (SELECT * FROM vectors ORDER BY root_id,x,y,z) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)', [str(temp)])
            temp.replace(archive)
        c.read_parquet(str(archive)).create_view('vectors', replace=True)
        for i, item in enumerate(p['source_inventory']):
            path = SOURCE / item['path']
            if f.sha(path) != item['sha256']:
                raise ValueError(f'Source dataset changed: {path}')
            with np.load(path) as z:
                arrays = {key:z[key] for key in z.files}
            arrays['x'] = f.vectors_at(c, int(item['root_id']), arrays['xyz_nm']) if item['rows'] else np.empty((0,64),np.float32)
            dest = output / item['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(temp, **arrays)
            with np.load(temp) as actual, np.load(path) as original:
                for key in original.files:
                    np.testing.assert_array_equal(actual[key], arrays['x'] if key == 'x' else original[key])
            temp.replace(dest)
            audit.append(dict(path=item['path'], sha256=f.sha(dest), rows=item['rows']))
            if i % 250 == 0:
                print('Assembled and verified', i, '/', len(p['source_inventory']), flush=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    manifest.update(variants=['geodesic'], derived_from=str(SOURCE), soma_embeddings=str(output/'soma_cells'),
                    embedding_run_id=f.RUN, embedding_checkpoint_id=f.CHECKPOINT,
                    embedding_checkpoint_sha256=p['checkpoint_sha256'], feature_plan=str(WORK/'plan.json'))
    f.save(output/'manifest.json', manifest)
    f.save(WORK/'data_verification.json', dict(complete=True, rows=p['sample_rows'], points=p['points'], files=audit,
           invariant='Every non-feature array equals the previous experiment exactly; all features finite float32[64]',
           checkpoint_sha256=p['checkpoint_sha256']))
    print('DATA READY', p['sample_rows'], flush=True)


def train(fold):
    import numpy as np
    p = check_plan()
    receipt = json.loads((WORK/'data_verification.json').read_text())
    if not receipt['complete'] or receipt['checkpoint_sha256'] != p['checkpoint_sha256']:
        raise ValueError('Dataset verification missing')
    os.environ['SUBCOMPARTMENT_DATA_DIR'] = str(WORK/'data')
    sys.path.insert(0, str(HERE))
    import train_compartment_gpu as gpu
    folds = json.loads((WORK/'folds.json').read_text())
    original = gpu.load
    def load(variant):
        x, y, roots, old_train, root_ids = original(variant)
        if set(root_ids) != set(folds['assignments']):
            raise ValueError('Cell inventory changed')
        mask = np.asarray([folds['assignments'][r] != fold for r in root_ids])[roots]
        expected = folds['summaries'][fold]
        for take, name in [(mask,'train'), (~mask,'test')]:
            np.testing.assert_array_equal(np.bincount(y[take],minlength=4), expected[f'{name}_counts'])
        return x, y, roots, mask, root_ids
    gpu.load = load
    gpu.BASE = WORK/'models'/f'fold{fold}'
    gpu.run('geodesic', epochs=100, head='resnet', sampling='inverse_sqrt')
    folder = gpu.BASE/'models_resnet_inverse_sqrt_100epochs/geodesic'
    checkpoint = gpu.torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    metadata = dict(fold=fold, n_folds=5, embedding_run_id=f.RUN, embedding_checkpoint_id=f.CHECKPOINT,
                    embedding_checkpoint_sha256=p['checkpoint_sha256'], feature_plan=str(WORK/'plan.json'),
                    selection_split=f'held-out cell fold {fold}', test_policy=folds['selection_policy'])
    checkpoint.update(metadata)
    gpu.torch.save(checkpoint, folder/'model.pt')
    for name in ['training.json','metrics.json']:
        value = json.loads((folder/name).read_text()); value.update(metadata); f.save(folder/name, value)
    f.save(folder/'complete.json', dict(complete=True, model_sha256=f.sha(folder/'model.pt'), **metadata))


def summarize():
    import numpy as np
    p = check_plan()
    results = []
    for fold in range(5):
        folder = WORK/'models'/f'fold{fold}'/'models_resnet_inverse_sqrt_100epochs/geodesic'
        complete = json.loads((folder/'complete.json').read_text())
        if not complete['complete'] or complete['model_sha256'] != f.sha(folder/'model.pt'):
            raise ValueError('Incomplete fold checkpoint')
        m = json.loads((folder/'metrics.json').read_text())
        results.append(dict(fold=fold, best_epoch=m['best_epoch'], macro_f1=m['macro_f1'], balanced_accuracy=m['balanced_accuracy'],
                            per_class={k:{q:m['classification_report'][k][q] for q in ['precision','recall','f1-score','support']}
                                       for k in ['soma','dendrite','axon','astrocytic_process']}))
    f.save(WORK/'complete.json', dict(complete=True, run_id=f.RUN, checkpoint_sha256=p['checkpoint_sha256'], folds=results,
           mean_macro_f1=float(np.mean([r['macro_f1'] for r in results])), precision_zero_division=0,
           evaluation='Cell-disjoint held-out folds; each fold also selects its best epoch, so scores are selection scores'))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan','smoke','embed','assemble','train','summarize'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=4)
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    if args.stage == 'plan': plan()
    elif args.stage == 'assemble': assemble()
    elif args.stage == 'train':
        if args.fold is None: parser.error('--fold is required')
        train(args.fold)
    elif args.stage == 'summarize': summarize()
    else:
        check_plan()
        f.smoke() if args.stage == 'smoke' else f.embed(args.rank, args.world)
