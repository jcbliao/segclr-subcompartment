"""Classify every canonical embedding in the completed historical-v117 cohort."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('SUBCOMPARTMENT_REPO', '/orcd/home/002/jcbliao/rotation/segclr/subcompartment_classification'))
sys.path.insert(0, str(REPO))
from segclr_db import store as st
from segclr_db import schema as S
from segclr_db.identity import LabelHierarchy, PredictionRun
from segclr_db.writer import SegCLRWriter
from subcompartment.model import load_model

FAMILY = os.environ.get('HISTORICAL_SUBCOMPARTMENT_FAMILY', 'six_class')
WORK = Path(os.environ.get('HISTORICAL_SUBCOMPARTMENT_WORK', f'/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/historical117_subcompartment_{FAMILY}_v1'))
DB = '/orcd/compute/sdorkenw/001/segclr-db'
NODESETS = {'l2': 'chunkedgraph_l2_v117', 'cave': 'cave_skeleton_v4'}
SHARDS = 64


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.chmod(0o664); tmp.replace(path)
def plan(): return read(WORK/'plan.json')
def log(**values): print(json.dumps(values), flush=True)


def prepare(inventory):
    import torch
    if (WORK/'plan.json').exists():
        log(stage='prepared_already', work=str(WORK)); return
    audit = read(inventory); source = audit['source']; ready = audit['ready']
    roots = sorted(map(int, source['roots']))
    if len(set(roots)) != len(roots) or set(map(str, roots)) != set(ready): raise ValueError('Source root inventory differs')
    models = []
    manifest = read(REPO/'models/manifest.json')[FAMILY]
    classes = None
    for entry in manifest['models']:
        src = REPO/entry['file']
        if sha(src) != entry['sha256']: raise ValueError('Model hash mismatch')
        cp = torch.load(src, weights_only=True, map_location='cpu')
        metadata = read(src.with_name(src.stem+'_training.json'))
        if metadata['embedding_checkpoint_sha256'] != source['checkpoint_sha256']: raise ValueError('Wrong encoder weights')
        if cp['embedding_run_id'] != source['run_id'] or cp['embedding_checkpoint_id'] != source['checkpoint_id']: raise ValueError('Wrong encoder identity')
        if classes is None: classes = cp['classes']
        if cp['classes'] != classes or cp['fold'] != entry['fold']: raise ValueError('Wrong model vocabulary/fold')
        dst = WORK/'models'/src.name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src,dst)
        models.append(dict(path=str(dst), sha256=sha(dst), fold=entry['fold'], best_epoch=cp['best_epoch'], classes=classes))
    # Balance complete roots; no root crosses rank boundaries in either node set.
    loads = np.zeros(SHARDS, np.int64); assignments = {}; counts = []
    for root in sorted(roots, key=lambda r: -(ready[str(r)]['l2_nodes'] + ready[str(r)]['skeleton_vertices'])):
        rank = int(loads.argmin()); row = ready[str(root)]
        assignments[str(root)] = rank; loads[rank] += row['l2_nodes'] + row['skeleton_vertices']
    for root in roots:
        row=ready[str(root)]
        counts.append(dict(root_id=root, rank=assignments[str(root)], l2=row['l2_nodes'], cave=row['skeleton_vertices']))
    hierarchy = LabelHierarchy.from_tree('subcompartment_'+FAMILY+'_historical117_v1', {c:None for c in classes})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    for model in models:
        model['experiment_id'] = f'subcompartment_{FAMILY}_hist117_fold{model["fold"]}_{model["sha256"][:12]}'
        model['classifier_run_id'] = model['experiment_id']+'__'+stamp
        model['prediction_runs'] = {kind:model['classifier_run_id']+'.best.'+kind for kind in NODESETS}
    value=dict(version=1, family=FAMILY, models=models, classes=classes, logit_classes=list(hierarchy.level_classes[0]),
        hierarchy_id=hierarchy.hierarchy_id, stamp=stamp, node_sets=NODESETS, shards=SHARDS, roots=counts,
        source_plan=source, source_snapshot=audit['snapshot'], source_inventory_sha256=sha(inventory),
        expected=dict(l2=source['l2_nodes'],cave=source['skeleton_vertices']),
        embedding_run_id=source['run_id'],embedding_checkpoint_id=source['checkpoint_id'],
        embedding_checkpoint_sha256=source['checkpoint_sha256'],
        input_policy='Exact canonical node_embeddings snapshot; preserve historical root/node-set/native node IDs; all roots and all nodes, no training-mask filtering.',
        scoring_policy='Five independent folds, raw logits and predictive entropy; no averaging.')
    save(WORK/'source_inventory.json',audit); save(WORK/'plan.json',value)
    log(stage='prepared',family=FAMILY,roots=len(roots),expected=value['expected'],snapshot=audit['snapshot'])


def register():
    p=plan(); digest=sha(WORK/'plan.json')
    if (WORK/'registered.json').exists():
        if read(WORK/'registered.json')['plan_sha256']!=digest: raise ValueError('Registration plan changed')
        return
    store=st.open_store(DB,'microns');st.maybe_compact=lambda *a,**k:False
    writer=SegCLRWriter(store=store)
    hierarchy=LabelHierarchy.from_tree(p['hierarchy_id'],{c:None for c in p['classes']});writer.register_hierarchy(hierarchy)
    for model in p['models']:
        checkpoint_id=f"checkpoint_e{model['best_epoch']}_s0"
        existing=st.scan(store,'runs',filter=f"run_id='{model['classifier_run_id']}'").to_pylist()
        if existing:
            writer.register_checkpoints(model['classifier_run_id'],[(checkpoint_id,model['best_epoch'],0,model['path'])])
        writer.register_experiment(model['experiment_id'],dict(head='resnet',classes=p['classes'],fold=model['fold'],model_sha256=model['sha256'],source_snapshot=p['source_snapshot'],historical_materialization=117),kind='classifier',hierarchy_id=p['hierarchy_id'])
        run_dir=Path(model['path']).parent if model['fold']==0 else WORK/'registered_models'/f"fold{model['fold']}"
        run_dir.mkdir(parents=True,exist_ok=True)
        run=writer.register_run(model['experiment_id'],run_dir,p['stamp'],seed=0)
        if run.run_id!=model['classifier_run_id']: raise ValueError('Run identity differs')
        writer.register_checkpoints(run.run_id,[(checkpoint_id,model['best_epoch'],0,model['path'])])
        for kind,ns in p['node_sets'].items():
            writer.register_prediction_run(PredictionRun(classifier_run_id=run.run_id,classifier_checkpoint_id=checkpoint_id,
                embedding_run_id=p['embedding_run_id'],embedding_checkpoint_id=p['embedding_checkpoint_id'],
                agg_spec_id=None,window_nm=None,prediction_run_id=model['prediction_runs'][kind],
                notes=json.dumps(dict(plan=str(WORK/'plan.json'),plan_sha256=digest,node_set_id=ns,scope=p['input_policy'],scoring=p['scoring_policy']))))
    if not (WORK/'publication_baseline.json').exists():
        baseline={}
        for name in ['predictions','prediction_logits']:
            ds=st.open_table(store,name) if st.table_exists(store,name) else None
            baseline[name]=dict(version=int(ds.version) if ds else None,fragment_ids=[int(f.fragment_id) for f in ds.get_fragments()] if ds else [])
        save(WORK/'publication_baseline.json',baseline)
    save(WORK/'import_migration.json',dict(complete=True,plan_sha256=digest,legacy_groups={},fresh_scope_proof='New timestamped runs registered before any prediction writes; one publisher per fold with disjoint root/rank groups.'))
    save(WORK/'prediction_runs.json',[dict(fold=m['fold'],**m['prediction_runs']) for m in p['models']])
    save(WORK/'registered.json',dict(complete=True,plan_sha256=digest))
    log(stage='registered',runs=10)


def stage():
    import lance
    p=plan();digest=sha(WORK/'plan.json')
    if (WORK/'inputs_complete.json').exists():
        if read(WORK/'inputs_complete.json')['plan_sha256']!=digest:raise ValueError('Input plan changed')
        return
    directory=WORK/'inputs'; directory.mkdir(exist_ok=True)
    with (WORK/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A failed staging pass may be rerun from the same frozen snapshot.
        # Only this stage's own partial artifacts are replaced before its completion marker.
        for path in directory.glob('*/*/*.parquet'):path.unlink()
        for path in directory.glob('*/*/*.json'):path.unlink()
        roots=np.array([r['root_id'] for r in p['roots']],np.int64)
        ranks=np.array([r['rank'] for r in p['roots']],np.int16)
        counts={kind:np.array([r[kind] for r in p['roots']],np.int64) for kind in NODESETS}
        offsets={kind:np.r_[0,np.cumsum(counts[kind])[:-1]] for kind in NODESETS}
        seen={kind:np.zeros(p['expected'][kind],bool) for kind in NODESETS}
        buffers=defaultdict(list);sizes=defaultdict(int);files=defaultdict(list);totals=defaultdict(int)
        def flush(key):
            if not sizes[key]:return
            kind,rank=key;table=pa.concat_tables(buffers[key]);path=directory/kind/f'{rank:02d}'/f'{len(files[key]):05d}.parquet'
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.parquet')
            pq.write_table(table,tmp,compression='zstd',compression_level=1);tmp.replace(path)
            receipt=dict(complete=True,path=str(path),rows=len(table),sha256=sha(path),plan_sha256=digest)
            save(path.with_suffix('.json'),receipt);files[key].append(receipt);buffers[key]=[];sizes[key]=0
        ds=lance.dataset(p['source_snapshot']['uri'],version=p['source_snapshot']['version'])
        scope=(f"run_id='{p['embedding_run_id']}' AND checkpoint_id='{p['embedding_checkpoint_id']}' "
               "AND node_set_id IN ('chunkedgraph_l2_v117','cave_skeleton_v4') AND root_id IN ("+','.join(map(str,roots))+')')
        reader=ds.scanner(columns=['root_id','node_id','node_set_id','embedding'],filter=scope,
                          batch_size=65536,batch_readahead=2,fragment_readahead=2,io_buffer_size=128*2**20).to_reader()
        began=time.monotonic();last=began
        for batch in reader:
            table=pa.Table.from_batches([batch])
            for kind,ns in NODESETS.items():
                part=table.filter(pc.equal(table['node_set_id'],ns))
                if not len(part):continue
                ids=np.asarray(part['root_id'],np.int64);nodes=np.asarray(part['node_id'],np.int64)
                pos=np.searchsorted(roots,ids)
                if (pos>=len(roots)).any() or not np.array_equal(roots[pos],ids):raise ValueError('Unexpected source root')
                if (nodes<0).any() or (nodes>=counts[kind][pos]).any():raise ValueError('Invalid source node identity')
                keys=offsets[kind][pos]+nodes
                if len(np.unique(keys))!=len(keys) or seen[kind][keys].any():raise ValueError('Duplicate source embedding identity')
                x=np.asarray(part['embedding'].combine_chunks().values).reshape(len(part),64)
                if not np.isfinite(x).all():raise ValueError('Nonfinite source embeddings')
                seen[kind][keys]=True;totals[kind]+=len(part)
                part=part.select(['root_id','node_id','embedding'])
                for rank in np.unique(ranks[pos]):
                    selected=part.filter(pa.array(ranks[pos]==rank));key=(kind,int(rank))
                    buffers[key].append(selected);sizes[key]+=len(selected)
                    if sizes[key]>=65536:flush(key)
            if time.monotonic()-last>20:
                log(stage='staging',rows=dict(totals),seconds=round(time.monotonic()-began,1));last=time.monotonic()
                save(WORK/'stage_progress.json',dict(rows=dict(totals),seconds=time.monotonic()-began))
        for kind in NODESETS:
            if not seen[kind].all() or totals[kind]!=p['expected'][kind]:raise ValueError(f'Missing source embeddings: {kind}')
            for rank in range(p['shards']):
                key=(kind,rank);flush(key)
                save(WORK/'status'/f'select_{kind}_{rank:02d}.json',dict(complete=True,plan_sha256=digest,nodes=sum(f['rows'] for f in files[key]),files=files[key]))
        save(WORK/'inputs_complete.json',dict(complete=True,plan_sha256=digest,rows=dict(totals),unique_identity_verified=True,seconds=time.monotonic()-began))
        log(stage='staged',rows=dict(totals),seconds=time.monotonic()-began)


def model_bundle(p, device):
    import copy
    import torch
    from torch.func import functional_call, stack_module_state
    models=[]
    for m in p['models']:
        if sha(m['path'])!=m['sha256']:raise ValueError('Model changed')
        model,metadata=load_model(m['path'],device)
        if metadata['classes']!=p['classes']:raise ValueError('Model class order changed')
        models.append(model)
    params,buffers=stack_module_state(models);base=copy.deepcopy(models[0]).to('meta')
    def call(a,b,x):return functional_call(base,(a,b),(x,))
    vmapped=torch.vmap(call,in_dims=(0,0,None))
    order=torch.tensor([p['classes'].index(c) for c in p['logit_classes']],device=device)
    def score(x):return vmapped(params,buffers,x)[:,:,order]
    return models,score


def pilot(path, device='cuda'):
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    p=plan();models,score=model_bundle(p,device)
    data=pq.read_table(path)
    x=np.asarray(data['embedding'].combine_chunks().values,np.float32).reshape(len(data),64)
    x=torch.from_numpy(x.copy()).to(device)
    order=[p['classes'].index(c) for c in p['logit_classes']]
    with torch.inference_mode():
        got=score(x);expected=torch.stack([m(x)[:,order] for m in models])
    torch.testing.assert_close(got,expected,rtol=3e-5,atol=3e-5)
    if not torch.equal(got.argmax(2),expected.argmax(2)):raise ValueError('Vectorized labels differ')
    if not torch.isfinite(got).all():raise ValueError('Nonfinite pilot scores')
    # Compare deployed CUDA scores with independent CPU checkpoint loading.
    cpu=[]
    for m in p['models']:
        model,_=load_model(m['path'])
        with torch.inference_mode():cpu.append(model(x.cpu())[:,order])
    cpu=torch.stack(cpu)
    torch.testing.assert_close(got.cpu(),cpu,rtol=3e-4,atol=3e-4)
    if not torch.equal(got.cpu().argmax(2),cpu.argmax(2)):raise ValueError('CPU/GPU pilot labels differ')
    result=dict(complete=True,plan_sha256=sha(WORK/'plan.json'),rows=len(data),folds=len(models),
                classes=p['logit_classes'],device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,
                max_cpu_gpu_abs_difference=float((got.cpu()-cpu).abs().max()),vectorized_label_parity=True)
    save(WORK/f'pilot_{device}.json',result);log(stage='pilot',**result)


def score_file(path, kind, rank, p, score, device='cuda'):
    import torch
    digest=sha(WORK/'plan.json');source=read(path.with_suffix('.json'))
    output=WORK/'scores'/kind/f'{rank:02d}'/path.name;receipt=output.with_suffix('.json')
    if receipt.exists():
        old=read(receipt)
        if old['input_sha256']!=source['sha256'] or old['plan_sha256']!=digest or sha(output)!=old['sha256']:raise ValueError('Invalid existing scores')
        return old
    if source['plan_sha256']!=digest or sha(path)!=source['sha256']:raise ValueError('Input artifact changed')
    table=pq.read_table(path);n=len(table)
    x=np.asarray(table['embedding'].combine_chunks().values,np.float32).reshape(n,64)
    logits=np.empty((len(p['models']),n,len(p['classes'])),np.float32)
    with torch.inference_mode():
        for start in range(0,n,8192):
            logits[:,start:start+8192]=score(torch.from_numpy(x[start:start+8192].copy()).to(device)).cpu().numpy()
    if not np.isfinite(logits).all():raise ValueError('Nonfinite inference')
    values={'root_id':table['root_id'],'node_id':table['node_id']}
    for fold in range(len(p['models'])):
        values[f'fold{fold}_logits']=pa.FixedSizeListArray.from_arrays(pa.array(logits[fold].ravel()),len(p['classes']))
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.tmp.parquet')
    pq.write_table(pa.table(values),tmp,compression='zstd',compression_level=1)
    back=pq.read_table(tmp)
    if not back['root_id'].equals(table['root_id']) or not back['node_id'].equals(table['node_id']):raise ValueError('Output identity changed')
    for fold in range(len(p['models'])):
        np.testing.assert_array_equal(np.asarray(back[f'fold{fold}_logits'].combine_chunks().values).reshape(n,-1),logits[fold])
    tmp.replace(output)
    result=dict(complete=True,path=str(output),rows=n,sha256=sha(output),input_sha256=source['sha256'],plan_sha256=digest)
    save(receipt,result);return result


def infer_worker(worker, device='cuda'):
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    p=plan();digest=sha(WORK/'plan.json')
    pilot_receipt=read(WORK/f'pilot_{device}.json')
    if not pilot_receipt['complete'] or pilot_receipt['plan_sha256']!=digest:raise ValueError('Pilot is not valid')
    models,score=model_bundle(p,device);claims=WORK/'claims';claims.mkdir(exist_ok=True)
    started=time.monotonic();completed=0
    # Individual immutable files are the shared-pool units. No static GPU partition.
    while True:
        tasks=[]
        for kind in NODESETS:
            for receipt in (WORK/'inputs'/kind).glob('*/*.json'):
                rank=int(receipt.parent.name)
                if not (WORK/'scores'/kind/receipt.parent.name/receipt.name).exists():tasks.append((kind,rank,receipt.with_suffix('.parquet')))
        tasks.sort(key=lambda x:(-x[2].stat().st_size,str(x[2])))
        progress=False
        for kind,rank,path in tasks:
            with (claims/f'{kind}_{rank:02d}_{path.stem}.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (WORK/'scores'/kind/f'{rank:02d}'/path.with_suffix('.json').name).exists():continue
                score_file(path,kind,rank,p,score,device);completed+=1;progress=True
                if completed%10==0:log(stage='inference',worker=worker,files=completed,seconds=time.monotonic()-started)
        if not tasks and (WORK/'inputs_complete.json').exists():break
        if not progress:time.sleep(3)
    # Every rank marker contains exactly its selected files, including empty ranks.
    for kind in NODESETS:
        for rank in range(p['shards']):
            selected=read(WORK/'status'/f'select_{kind}_{rank:02d}.json')
            entries=[read(WORK/'scores'/kind/f'{rank:02d}'/Path(f['path']).with_suffix('.json').name) for f in selected['files']]
            if sum(f['rows'] for f in entries)!=selected['nodes']:raise ValueError('Scored selection differs')
            save(WORK/'status'/f'infer_{kind}_{rank:02d}.json',dict(complete=True,plan_sha256=digest,nodes=selected['nodes'],files=entries))
    save(WORK/'status'/f'gpu_worker_{worker}.json',dict(complete=True,files=completed,seconds=time.monotonic()-started,plan_sha256=digest))
    log(stage='worker_complete',worker=worker,files=completed)


def expected_tables(table, fold, run, node_set, classes):
    n=len(table);raw=np.asarray(table[f'fold{fold}_logits'].combine_chunks().values,np.float32).reshape(n,len(classes))
    if not np.isfinite(raw).all():raise ValueError('Invalid logits')
    # NumPy Unicode conversion can yield a ChunkedArray for production-sized
    # groups. Arrow take keeps the small class vocabulary and returns an Array.
    labels=pa.array(classes).take(pa.array(raw.argmax(1),pa.int64()))
    probs=np.exp(raw-raw.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True)
    entropy=-(probs*np.log(np.maximum(probs,np.float32(1e-30)))).sum(1).astype(np.float32)
    common=dict(prediction_run_id=pa.array([run]*n),root_id=table['root_id'].cast(pa.int64()),
                node_id=table['node_id'].cast(pa.int32()),synapse_id=pa.nulls(n,pa.int64()),node_set_id=pa.array([node_set]*n))
    predictions=pa.table(dict(**common,subject_root_id=common['root_id'],
        pred_labels=pa.ListArray.from_arrays(pa.array(np.arange(n+1),pa.int32()),labels),
        pred_uncertainty=pa.ListArray.from_arrays(pa.array(np.arange(n+1),pa.int32()),pa.array(entropy))))
    logits=pa.table(dict(**common,level=pa.array(np.zeros(n,np.int8)),
        logits=pa.ListArray.from_arrays(pa.array(np.arange(n+1)*len(classes),pa.int32()),pa.array(raw.ravel()))))
    return [predictions.select(S.get('predictions').schema.names).cast(S.get('predictions').schema),
            logits.select(S.get('prediction_logits').schema.names).cast(S.get('prediction_logits').schema)]


def import_fold(fold):
    from historical_import_support import _attempt_ledger,_scan_attempt,missing_rows
    p=plan();digest=sha(WORK/'plan.json');store=st.open_store(DB,'microns');st.maybe_compact=lambda *a,**k:False
    if read(WORK/'registered.json')['plan_sha256']!=digest:raise ValueError('Wrong registry plan')
    summary=defaultdict(int)
    for kind,node_set in NODESETS.items():
        for rank in range(p['shards']):
            receipt=WORK/'imports'/f'fold{fold}'/kind/f'{rank:02d}.json'
            receipt.parent.mkdir(parents=True,exist_ok=True)
            with receipt.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                selected=read(WORK/'status'/f'select_{kind}_{rank:02d}.json')
                scored=read(WORK/'status'/f'infer_{kind}_{rank:02d}.json')
                if scored['plan_sha256']!=digest or not scored['complete'] or selected['nodes']!=scored['nodes']:raise ValueError('Invalid score status')
                identity=[dict(path=f['path'],sha256=f['sha256'],rows=f['rows']) for f in scored['files']]
                group_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                if receipt.exists():
                    old=read(receipt)
                    if old['plan_sha256']!=digest or old['group_sha256']!=group_hash or not old['complete']:raise ValueError('Invalid import receipt')
                    summary[kind]+=old['rows'];continue
                if not identity:
                    save(receipt,dict(complete=True,rows=0,plan_sha256=digest,group_sha256=group_hash));continue
                tables=[]
                for source,item in zip(selected['files'],scored['files'],strict=True):
                    if item['input_sha256']!=source['sha256'] or item['rows']!=source['rows'] or sha(item['path'])!=item['sha256']:raise ValueError('Invalid score artifact')
                    tables.append(pq.read_table(item['path'],columns=['root_id','node_id',f'fold{fold}_logits']))
                table=pa.concat_tables(tables);n=len(table)
                if n!=selected['nodes']:raise ValueError('Group row count changed')
                roots=np.unique(np.asarray(table['root_id'],np.int64))
                run=p['models'][fold]['prediction_runs'][kind]
                scope=f"prediction_run_id='{run}' AND node_set_id='{node_set}' AND root_id BETWEEN {int(roots[0])} AND {int(roots[-1])}"
                for name,wanted in zip(['predictions','prediction_logits'],expected_tables(table,fold,run,node_set,p['logit_classes'])):
                    attempt=_attempt_ledger(store,name,scope,roots,wanted,(fold,kind,rank,group_hash),digest)
                    actual=_scan_attempt(store,name,attempt,roots,wanted.schema)
                    fresh=missing_rows(wanted,actual,name)
                    if len(fresh):st.append(store,name,fresh)
                    after=_scan_attempt(store,name,attempt,roots,wanted.schema)
                    if len(after)!=n or len(missing_rows(wanted,after,name)):raise ValueError('Exact database readback failed')
                save(receipt,dict(complete=True,rows=n,plan_sha256=digest,group_sha256=group_hash,prediction_run_id=run,node_set_id=node_set,exact_db_roundtrip=True))
                summary[kind]+=n;log(stage='imported',fold=fold,kind=kind,rank=rank,rows=n)
    if dict(summary)!=p['expected']:raise ValueError('Incomplete fold publication')
    save(WORK/'status'/f'import_fold{fold}.json',dict(complete=True,plan_sha256=digest,rows=dict(summary)))


def finalize():
    p=plan();digest=sha(WORK/'plan.json')
    if read(WORK/'inputs_complete.json')['plan_sha256']!=digest:raise ValueError('Input plan differs')
    for fold in range(len(p['models'])):
        receipt=read(WORK/'status'/f'import_fold{fold}.json')
        if not receipt['complete'] or receipt['plan_sha256']!=digest or receipt['rows']!=p['expected']:raise ValueError('Fold import incomplete')
    store=st.open_store(DB,'microns');baseline=read(WORK/'publication_baseline.json')
    roots=np.array([r['root_id'] for r in p['roots']],np.int64)
    counts={k:np.array([r[k] for r in p['roots']],np.int64) for k in NODESETS}
    offsets={k:np.r_[0,np.cumsum(counts[k])[:-1]] for k in NODESETS}
    runs={m['prediction_runs'][k]:(m['fold'],k) for m in p['models'] for k in NODESETS}
    audits={}
    for name in ['predictions','prediction_logits']:
        seen={run:np.zeros(p['expected'][kind],bool) for run,(_,kind) in runs.items()}
        ds=st.open_table(store,name);fragments=ds.get_fragments();base=set(baseline[name]['fragment_ids'])
        current={int(f.fragment_id) for f in fragments}
        selected=[f for f in fragments if int(f.fragment_id) not in base] if base.issubset(current) else None
        scope='prediction_run_id IN ('+','.join(st.sql_literal(run) for run in runs)+')'
        extra='subject_root_id' if name=='predictions' else 'level'
        reader=ds.scanner(columns=['prediction_run_id','root_id','node_id','node_set_id','synapse_id',extra],
            filter=scope,fragments=selected,use_scalar_index=False,batch_size=131072,
            batch_readahead=2,fragment_readahead=2,io_buffer_size=128*2**20).to_reader()
        total=0;began=time.monotonic();last=began
        for batch in reader:
            table=pa.Table.from_batches([batch])
            for run in pc.unique(table['prediction_run_id']).to_pylist():
                if run not in runs:raise ValueError('Unexpected prediction run')
                part=table.filter(pc.equal(table['prediction_run_id'],run));_,kind=runs[run]
                if pc.any(pc.not_equal(part['node_set_id'],NODESETS[kind])).as_py() or part['synapse_id'].null_count!=len(part):raise ValueError('Wrong node namespace/synapse identity')
                r=np.asarray(part['root_id'],np.int64);n=np.asarray(part['node_id'],np.int64);pos=np.searchsorted(roots,r)
                if (pos>=len(roots)).any() or not np.array_equal(roots[pos],r):raise ValueError('Unexpected predicted root')
                if (n<0).any() or (n>=counts[kind][pos]).any():raise ValueError('Invalid predicted node')
                keys=offsets[kind][pos]+n
                if len(np.unique(keys))!=len(keys) or seen[run][keys].any():raise ValueError('Duplicate prediction identity')
                if extra=='subject_root_id' and not np.array_equal(np.asarray(part[extra]),r):raise ValueError('Prediction subject differs')
                if extra=='level' and np.any(np.asarray(part[extra])!=0):raise ValueError('Unexpected logit level')
                seen[run][keys]=True;total+=len(part)
            if time.monotonic()-last>20:
                log(stage='auditing',table=name,rows=total,seconds=time.monotonic()-began);last=time.monotonic()
        if not all(a.all() for a in seen.values()):raise ValueError('Final audit found missing predictions')
        audits[name]=dict(rows=total,unique_identity_verified=True,all_expected_nodes=True,version=int(ds.version))
        del seen
    result=dict(complete=True,plan_sha256=digest,roots=len(roots),nodes=p['expected'],folds=len(p['models']),classes=p['logit_classes'],
        predictions=sum(p['expected'].values())*len(p['models']),audits=audits,prediction_runs=read(WORK/'prediction_runs.json'))
    save(WORK/'complete.json',result);log(stage='complete',**result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','register','stage','pilot','infer','import','finalize'])
    parser.add_argument('--inventory',default='/tmp/historical_subcompartment_inventory.json')
    parser.add_argument('--data',default='/tmp/historical_subcompartment_pilot.parquet')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--worker',default=os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    parser.add_argument('--fold',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.inventory)
    elif args.stage=='register':register()
    elif args.stage=='stage':stage()
    elif args.stage=='pilot':pilot(args.data,args.device)
    elif args.stage=='infer':infer_worker(args.worker,args.device)
    elif args.stage=='import':import_fold(args.fold)
    else:finalize()


if __name__=='__main__':main()
