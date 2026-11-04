"""Add glial process classes outside 10 um of verified database soma centers."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import duckdb
from six_class_data import WORK,BASE,CLASSES,write_json

HERE=Path(__file__).resolve().parent
GNN=HERE.parent/'gnn_classifier'
OLD=WORK.parent/'subcompartment_latest100k_geo10um_v1'
sys.path.insert(0,str(GNN))
from scripts import latest_segclr_features as f
from segclr_db import store as st
from segclr_db.skeletons import skeleton_content_hash


def hashes():
    return {str(p):f.sha(p) for p in [Path(__file__),HERE/'six_class_data.py',HERE/'train_compartment_six.py',GNN/'gnn/resnet.py']}


def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((GNN/'data/manifest.json').read_text())
    added={r:m for r,m in manifest['cells'].items() if m['cell_type'] in {'microglia','oligo'}}
    if len(added)!=46:raise ValueError('Source glia inventory changed')
    original=json.loads((OLD/'complete.json').read_text())
    if not original['complete'] or original['run_id']!=f.RUN:raise ValueError('Original run unavailable')
    store=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns')
    clause='root_id IN ('+','.join(added)+')'
    args=dict(use_scalar_index=False,batch_size=8192,batch_readahead=1,fragment_readahead=1)
    cells=st.open_table(store,'cells')
    centers={str(x['root_id']):[x[k] for k in ['soma_x_nm','soma_y_nm','soma_z_nm']] for x in cells.scanner(filter=clause,**args).to_table().to_pylist()}
    table=st.open_table(store,'node_manifest')
    registered=table.scanner(columns=['root_id','content_hash'],filter=clause+" AND node_set_id='cave_skeleton_v4'",**args).to_table().to_pylist()
    geometry_hashes={str(x['root_id']):x['content_hash'] for x in registered}
    versions=dict(cells=cells.version,node_manifest=table.version)
    audits=[]
    for folder in ['geodesic/cells','soma_cells']:(BASE/folder).mkdir(parents=True,exist_ok=True)
    with duckdb.connect(str(WORK/'glia.duckdb')) as c:
        c.execute('SET threads=4');c.execute("SET memory_limit='32GB'")
        c.execute('CREATE OR REPLACE TABLE requests(root_id BIGINT,node_id INTEGER,x FLOAT,y FLOAT,z FLOAT)')
        for rid,meta in sorted(added.items()):
            center=np.asarray(centers.get(rid),dtype=float)
            if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError(f'{rid}: valid soma center required')
            with (GNN/'data/skeleton_cache'/f'{rid}.pkl').open('rb') as stream:sk=pickle.load(stream)
            digest=skeleton_content_hash(sk)
            if digest!=geometry_hashes.get(rid):raise ValueError(f'{rid}: registered geometry hash differs')
            xyz=np.asarray(sk.coords,np.float32);distance=np.linalg.norm(xyz.astype(np.float64)-center,axis=1)
            ids=np.flatnonzero(distance>10000.).astype(np.int32)
            if not len(ids):raise ValueError(f'{rid}: no eligible points beyond soma exclusion')
            c.register('cell',pa.table(dict(root_id=np.repeat(int(rid),len(ids)),node_id=ids,x=xyz[ids,0],y=xyz[ids,1],z=xyz[ids,2])))
            c.execute('INSERT INTO requests SELECT * FROM cell');c.unregister('cell')
            audits.append(dict(root_id=rid,cell_type=meta['cell_type'],split=meta['split'],center_nm=center.tolist(),center_source='segclr_db cells.soma_*_nm',
                  geometry_sha256=digest,total_nodes=len(xyz),excluded_near_soma=int(np.sum(distance<=10000)),eligible_points=len(ids),min_distance_nm=float(distance[ids].min())))
        embeddings=st.open_table(store,'node_embeddings',64);versions['node_embeddings']=embeddings.version
        c.register('embeddings',embeddings.scanner(columns=['root_id','node_id','embedding'],filter=clause+f" AND run_id='{f.RUN}' AND checkpoint_id='{f.CHECKPOINT}' AND node_set_id='cave_skeleton_v4'",**args).to_reader())
        print('Read newest glial embeddings from segclr_db',flush=True)
        c.execute('CREATE OR REPLACE TABLE raw AS SELECT e.* FROM embeddings e JOIN requests r USING(root_id,node_id)')
        conflicts=c.execute('SELECT count(*) FROM (SELECT root_id,node_id FROM raw GROUP BY 1,2 HAVING count(DISTINCT embedding)>1)').fetchone()[0]
        if conflicts:raise ValueError('Conflicting database vectors')
        c.execute('CREATE OR REPLACE TABLE vectors AS SELECT root_id,node_id,first(embedding) AS embedding FROM raw WHERE embedding IS NOT NULL AND len(embedding)=64 AND isfinite(list_sum(embedding)) GROUP BY 1,2')
        missing=c.execute('SELECT count(*) FROM requests ANTI JOIN vectors USING(root_id,node_id)').fetchone()[0]
        write_json(WORK/'coverage.json',dict(missing_points=missing,versions=versions,run_id=f.RUN,checkpoint_id=f.CHECKPOINT,cells=audits))
        if missing:raise ValueError(f'{missing} requested glial points lack database embeddings; resolve before training')
        for item in audits:
            rid=item['root_id'];cls=4 if item['cell_type']=='microglia' else 5
            t=c.execute('SELECT r.*,v.embedding FROM requests r JOIN vectors v USING(root_id,node_id) WHERE root_id=? ORDER BY node_id',[int(rid)]).to_arrow_table()
            x=np.asarray(t['embedding'].combine_chunks().values,np.float32).reshape(-1,64)
            ids=np.asarray(t['node_id']);xyz=np.column_stack([np.asarray(t[k]) for k in ['x','y','z']])
            assert np.all(np.linalg.norm(xyz.astype(float)-item['center_nm'],axis=1)>10000)
            np.savez_compressed(BASE/'geodesic/cells'/f'{rid}.npz',x=x,y=np.full(len(x),cls,np.int8),point_id=ids,xyz_nm=xyz,root_id=rid,split=item['split'])
            print('Prepared',rid,item['cell_type'],len(x),flush=True)
        c.execute('CHECKPOINT')
    # The previously verified four-class arrays are reused byte for byte.
    for folder in ['soma_cells','geodesic/cells']:
        for src in sorted((OLD/'data'/folder).glob('*.npz')):
            dst=BASE/folder/src.name
            if dst.is_symlink():
                if dst.resolve()!=src.resolve():raise ValueError('Unexpected existing source link')
            elif dst.exists():raise ValueError('Refusing to replace existing data')
            else:dst.symlink_to(src.resolve())
    folds=json.loads((OLD/'folds.json').read_text())
    assignments=folds['assignments'].copy()
    rng=np.random.default_rng(0)
    for kind in ['microglia','oligo']:
        train=sorted(r for r,m in added.items() if m['cell_type']==kind and m['split']=='train');rng.shuffle(train)
        for i,rid in enumerate(train):assignments[rid]=1+i%4
        for rid,m in added.items():
            if m['cell_type']==kind and m['split']!='train':assignments[rid]=0
    counts={}
    for folder in ['soma_cells','geodesic/cells']:
        for path in sorted((BASE/folder).glob('*.npz')):
            with np.load(path) as z:
                rid=str(z['root_id']);y=z['y'];x=z['x']
                if x.shape!=(len(y),64) or not np.isfinite(x).all():raise ValueError('Invalid prepared features')
                counts.setdefault(rid,np.zeros(6,np.int64));counts[rid]+=np.bincount(y,minlength=6)
    if set(counts)!=set(assignments):raise ValueError('Fold/data inventories differ')
    summaries=[]
    for fold in range(5):
        test=[r for r in counts if assignments[r]==fold];train=[r for r in counts if assignments[r]!=fold]
        tr=sum((counts[r] for r in train),np.zeros(6,np.int64));te=sum((counts[r] for r in test),np.zeros(6,np.int64))
        assert np.all(tr>0) and np.all(te>0) and not set(test)&set(train)
        summaries.append(dict(fold=fold,n_train_cells=len(train),n_test_cells=len(test),train_counts=tr.tolist(),test_counts=te.tolist()))
    write_json(WORK/'folds.json',dict(assignments=assignments,summaries=summaries,original_fold_plan=str(OLD/'folds.json'),policy='Original cells retain fold; new manifest-test glia join fold 0; manifest-train glia stratified by type over folds 1-4, seed 0'))
    write_json(BASE/'manifest.json',dict(classes=CLASSES,embedding_run_id=f.RUN,embedding_checkpoint_id=f.CHECKPOINT,excluded_cell_types=['OPC'],excluded_cell=['864691136364187874'],added_cells=audits,nucleus_exclusion_nm=10000))
    write_json(WORK/'data_verification.json',dict(complete=True,classes=CLASSES,n_cells=len(counts),n_points=int(sum(map(np.sum,counts.values()))),class_counts=sum(counts.values(),np.zeros(6,np.int64)).tolist(),versions=versions,code_hashes=hashes(),checkpoint_sha256=f.sha(f.WEIGHTS),source_four_class_run=str(OLD)))
    print('SIX-CLASS DATA VERIFIED',summaries,flush=True)


def train(fold):
    verify=json.loads((WORK/'data_verification.json').read_text())
    if not verify['complete'] or verify['code_hashes']!=hashes() or verify['checkpoint_sha256']!=f.sha(f.WEIGHTS):raise ValueError('Prepared inputs or code changed')
    import train_compartment_six as gpu
    folds=json.loads((WORK/'folds.json').read_text());original=gpu.load
    def load(variant):
        x,y,roots,train,root_ids=original(variant)
        assert set(root_ids)==set(folds['assignments'])
        mask=np.asarray([folds['assignments'][r]!=fold for r in root_ids])[roots]
        for take,name in [(mask,'train'),(~mask,'test')]:np.testing.assert_array_equal(np.bincount(y[take],minlength=6),folds['summaries'][fold][f'{name}_counts'])
        return x,y,roots,mask,root_ids
    gpu.load=load;gpu.BASE=WORK/'models'/f'fold{fold}'
    gpu.run('geodesic',epochs=100,head='resnet',sampling='inverse_sqrt')
    folder=gpu.BASE/'models_resnet_inverse_sqrt_100epochs/geodesic'
    cp=gpu.torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    cp.update(fold=fold,embedding_run_id=f.RUN,embedding_checkpoint_id=f.CHECKPOINT,selection_split=f'held-out cell fold {fold}')
    gpu.torch.save(cp,folder/'model.pt')
    for name in ['training.json','metrics.json']:
        value=json.loads((folder/name).read_text())
        value.update(fold=fold,n_folds=5,selection_split=f'held-out cell fold {fold}',
                     test_policy='Each held-out cell fold is evaluated every epoch and selects its best macro F1 checkpoint',
                     embedding_run_id=f.RUN,embedding_checkpoint_id=f.CHECKPOINT,
                     embedding_checkpoint_sha256=verify['checkpoint_sha256'],fold_plan=str(WORK/'folds.json'))
        write_json(folder/name,value)
    write_json(folder/'complete.json',dict(complete=True,fold=fold,model_sha256=f.sha(folder/'model.pt')))


def summarize():
    results=[];cms=[]
    for fold in range(5):
        folder=WORK/'models'/f'fold{fold}'/'models_resnet_inverse_sqrt_100epochs/geodesic'
        receipt=json.loads((folder/'complete.json').read_text());assert receipt['model_sha256']==f.sha(folder/'model.pt')
        m=json.loads((folder/'metrics.json').read_text());results.append(dict(fold=fold,macro_f1=m['macro_f1'],balanced_accuracy=m['balanced_accuracy'],best_epoch=m['best_epoch'],per_class={k:m['classification_report'][k] for k in CLASSES}));cms.append(m['confusion_matrix'])
    cm=np.sum(cms,axis=0)
    write_json(WORK/'complete.json',dict(complete=True,classes=CLASSES,folds=results,mean_macro_f1=float(np.mean([r['macro_f1'] for r in results])),mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in results])),confusion_matrix=cm.tolist(),precision_zero_division=0,evaluation='Each held-out cell fold also selects the best epoch. Scores are selection scores.'))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','summarize']);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':
        if a.fold is None:p.error('--fold required')
        train(a.fold)
    else:summarize()
