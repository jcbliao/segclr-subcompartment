"""Exact historical pair audit after same-subcompartment endpoint filtering.

Reuses the completed historical audit's graphs, membership caches and sampling
weights. No graph vertices or edges are removed. Five folds remain independent.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse.csgraph import dijkstra

AUDIT_CODE=Path(os.environ.get('PAIR_AUDIT_CODE','/orcd/home/002/jcbliao/rotation/segclr/quantify_segclr_pairs/scripts'))
sys.path.insert(0,str(AUDIT_CODE));sys.path.insert(0,str(AUDIT_CODE.parent))
from distance_balance_math import rows, bins
from distance_uniform import graph_data
from exponential_distance import bin_importance
from skeleton_array import Experiment, cached
from recover_pair_errors import recovery_class
import pair_array

BASE=Path('/orcd/scratch/orcd/013/jcbliao/segclr_pair_audit')
SKELETON=BASE/'v117_skeleton_v4_20261019'
L2=BASE/'v117_distance_uniform20_20261017'
ORIGINAL=BASE/'v117_l2_20261017'
SCORES=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/historical117_subcompartment_six_class_v1')
WORK=Path(os.environ.get('PAIR_FILTER_WORK',str(BASE/'v117_subcompartment_filter_20261108')))
MODES=['chunkedgraph','skeleton_matched_l2','skeleton_native_nodes']
POLICIES=[('all_pairs','raw'),('uniform_distance',None),('exponential_15um',15),
          ('exponential_30um',30),('exponential_50um',50),('exponential_100um',100)]
GLIA={'astrocyte','microglia','oligo','OPC'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def savez(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)


class OfflineRunner(recovery_class(pair_array)):
    def __init__(self):
        self.run=ORIGINAL
    def api(self,*args,**kwargs):raise RuntimeError('Audit requires an uncached API request')
    def cache(self,path,fn):return cached(path)
    def cell_l2(self,sv):
        key=hashlib.sha256(sv.tobytes()).hexdigest()
        if getattr(self,'_cell_l2_key',None)!=key:
            folder=self.run/'cache/recovery_sv_to_l2'/key
            self._cell_l2_parents=np.concatenate([cached(folder/f'{i}.npz')['parents'] for i in range(0,len(sv),10000)])
            if len(self._cell_l2_parents)!=len(sv):raise ValueError('Incomplete membership mapping')
            self._cell_l2_key=key
        return self._cell_l2_parents


def prepare():
    if (WORK/'plan.json').exists():return
    source=read(SCORES/'plan.json');available={r['root_id'] for r in source['roots']}
    all_neurons=[r for r in read(L2/'cells.json') if r['casey_coarse'] not in GLIA]
    cells=[r for r in read(SKELETON/'cells.json') if r['casey_coarse'] not in GLIA]
    assert all(set(r['historical_roots'])<=available for r in cells)
    selected={r['root_1718'] for r in cells}
    excluded=[r for r in all_neurons if r['root_1718'] not in selected]
    write(WORK/'cells.json',cells);write(WORK/'excluded_cells.json',excluded)
    write(WORK/'plan.json',dict(source_scores=str(SCORES),source_plan_sha256=sha(SCORES/'plan.json'),
        skeleton=str(SKELETON),l2=str(L2),cells=len(cells),excluded_neurons=len(excluded),
        cells_sha256=sha(WORK/'cells.json'),classes=source['logit_classes'],folds=list(range(5)),
        modes=MODES,policies=[p for p,_ in POLICIES],threshold_nm=150000,
        endpoint_filter='Retain pairs with identical argmax class, independently per fold; retain all six classes.',
        weights='Original all-root distance-bin weights fixed BEFORE filtering, including exponential policies.',
        aggregation='Sum per-cell correct/incorrect exposures; cross-cell pairs can contribute to both cells.',
        membership='Original reference-1718 supervoxel membership fractions; reference nonmembership is the existing error proxy.',
        missing_policy='Same complete neuronal cohort for all geometries; missing skeletons are explicit exclusions.'))


def export_labels(rank):
    done=WORK/'label_status'/f'{rank:02d}.json'
    if done.exists():return
    plan=read(SCORES/'plan.json');assert sha(SCORES/'plan.json')==read(WORK/'plan.json')['source_plan_sha256']
    roots=[r for r in plan['roots'] if r['rank']==rank];out={};counts={}
    for kind in ['l2','cave']:
        values={r['root_id']:np.full((r[kind],5),255,np.uint8) for r in roots}
        status=read(SCORES/'status'/f'infer_{kind}_{rank:02d}.json')
        assert status['complete'] and status['plan_sha256']==sha(SCORES/'plan.json')
        for entry in status['files']:
            assert sha(entry['path'])==entry['sha256']
            t=pq.read_table(entry['path']);rids=np.asarray(t['root_id']);ids=np.asarray(t['node_id'])
            labels=np.stack([np.asarray(t[f'fold{f}_logits'].combine_chunks().values).reshape(-1,6).argmax(1) for f in range(5)],1)
            for root in np.unique(rids):
                mask=rids==root;node=ids[mask];target=values[int(root)]
                assert len(np.unique(node))==len(node) and (target[node]==255).all()
                target[node]=labels[mask]
        assert all((v<6).all() for v in values.values())
        counts[kind]=sum(len(v) for v in values.values())
        for root,v in values.items():out.setdefault(root,{})[kind]=v
    for root,value in out.items():savez(WORK/'labels'/f'{root}.npz',**value)
    write(done,dict(complete=True,rank=rank,nodes=counts,source_plan_sha256=sha(SCORES/'plan.json')))


def l2_histograms(graph,coords,p,labels,edges):
    """Baseline plus five filters, shape (6, correct/incorrect, distance-bin)."""
    out=np.zeros((6,2,len(edges)-1))
    for source,targets,distance in rows(graph,coords,np.flatnonzero(p>0),150000.):
        ix=bins(distance,edges);g=p[source]*p[targets];b=p[source]*(1-p[targets])
        out[0,0]+=np.bincount(ix,weights=g,minlength=out.shape[-1])
        out[0,1]+=np.bincount(ix,weights=b,minlength=out.shape[-1])
        match=labels[targets]==labels[source]
        for fold in range(5):
            keep=match[:,fold]
            out[fold+1,0]+=np.bincount(ix[keep],weights=g[keep],minlength=out.shape[-1])
            out[fold+1,1]+=np.bincount(ix[keep],weights=b[keep],minlength=out.shape[-1])
    out[:,0]/=2
    return out


def pooled_population(size,projection,p,labels):
    # Slot zero pools all classes; slots 1..30 are fold/class combinations.
    pp=np.zeros((31,size));qq=np.zeros_like(pp);sg=np.zeros_like(pp);sb=np.zeros_like(pp)
    indices=[projection]+[projection+size*labels[:,fold].astype(np.int64) for fold in range(5)]
    for fold,ix in enumerate(indices):
        length=size if fold==0 else size*6
        start=0 if fold==0 else 1+(fold-1)*6;stop=1 if fold==0 else start+6
        for dest,weight in [(pp,p),(qq,1-p),(sg,p*p),(sb,p*(1-p))]:
            dest[start:stop]=np.bincount(ix,weights=weight,minlength=length).reshape(-1,size)
    return pp,qq,sg,sb


def skeleton_histograms(graph,populations,edges):
    """Reuse distance rows for both endpoint populations; preserve self exclusion.

    Distinct L2 endpoints collapsed to one vertex may have different classes.
    Pool each class separately and subtract identical original endpoints only.
    """
    pooled=[pooled_population(graph.shape[0],*pop) for pop in populations]
    sources=np.flatnonzero(np.any([v[0][0]>0 for v in pooled],axis=0))
    targets=np.flatnonzero(np.any([v[0][0]+v[1][0]>0 for v in pooled],axis=0))
    result=np.zeros((len(populations),6,2,len(edges)-1))
    for start in range(0,len(sources),16):
        selected=sources[start:start+16]
        ds=dijkstra(graph,directed=False,indices=selected,limit=150000.)
        for source,row in zip(selected,ds):
            distance=row[targets];keep=np.isfinite(distance)&(distance<=150000.)
            target=targets[keep];ix=bins(distance[keep].astype(np.float32),edges)
            self_index=np.flatnonzero(target==source)
            for pop,(pp,qq,sg,sb) in enumerate(pooled):
                if pp[0,source]==0:continue
                g=pp[:,source,None]*pp[:,target];b=pp[:,source,None]*qq[:,target]
                g[:,self_index]-=sg[:,source,None];b[:,self_index]-=sb[:,source,None]
                g=np.concatenate([g[:1],g[1:].reshape(5,6,-1).sum(1)])
                b=np.concatenate([b[:1],b[1:].reshape(5,6,-1).sum(1)])
                for fold in range(6):
                    result[pop,fold,0]+=np.bincount(ix,weights=g[fold],minlength=result.shape[-1])
                    result[pop,fold,1]+=np.bincount(ix,weights=b[fold],minlength=result.shape[-1])
    result[:,:,0]/=2
    return result


def score_cell(row):
    started=time.monotonic();cell=int(row['root_1718']);path=WORK/'cells'/f'{cell}.json'
    if path.exists():return
    exp=Experiment(SKELETON);runner=OfflineRunner();exp._membership_runner=runner
    sv=np.unique(np.load(ORIGINAL/'cells'/str(cell)/'current_sv.npz')['sv']).astype(np.uint64)
    baseline=read(L2/'cells'/str(cell)/'result.json');l2_old={int(x['root_117']):x for x in baseline['root_details']}
    previous=json.loads(cached(exp.cell_path(cell))['json_utf8'].tobytes())
    old={int(x['root_117']):x for x in previous['root_details']}
    total=np.zeros((3,len(POLICIES),6,2));all_hist=[]
    for root in row['historical_roots']:
        root=int(root);folder=L2/'roots'/str(root)
        graph,coords,meta=graph_data(folder);full=np.load(folder/'full_nodes.npy').astype(np.uint64)
        active=np.load(folder/'active_indices.npy');nodes=np.load(folder/'nodes.npy').astype(np.uint64)
        np.testing.assert_array_equal(full[active],nodes)
        anchor,denom=runner.memberships(Path(meta['source_directory']),full,sv)
        anchor=anchor[active];denom=denom[active];p=anchor/denom
        labels=dict(np.load(WORK/'labels'/f'{root}.npz'))
        assert len(labels['l2'])==len(p)
        edges=np.load(folder/'bin_edges_nm.npy')
        lh=l2_histograms(graph,coords,p,labels['l2'],edges)
        sg,d=exp.geometry(root);projection=d['projection'];occupied=d['occupied']
        assert len(labels['cave'])==sg.shape[0] and len(projection)==len(p)
        group_nodes=np.flatnonzero(np.isin(projection,np.unique(projection[anchor>0])))
        sizes=np.zeros(len(active),np.int64)
        sizes[group_nodes]=exp.group_sizes(Path(meta['source_directory']),full,active[group_nodes])
        nt=np.bincount(projection,weights=sizes,minlength=sg.shape[0])
        na=np.bincount(projection,weights=anchor,minlength=sg.shape[0])
        native=np.divide(na[occupied],nt[occupied],out=np.zeros(len(occupied)),where=nt[occupied]>0)
        sh=skeleton_histograms(sg,[(projection,p,labels['l2']),(occupied,native,labels['cave'][occupied])],edges)
        hist=np.concatenate([lh[None],sh]);all_hist.append(hist)
        for mode_i,mode in enumerate(MODES):
            ref=old[root]['variants'][mode]
            expected=np.array([ref['correct_weights'],ref['incorrect_weights']])
            np.testing.assert_allclose(hist[mode_i,0],expected,rtol=1e-8,atol=1e-5,err_msg=f'{cell}/{root}/{mode} baseline differs')
            assert np.all(hist[mode_i,1:]>=-1e-5) and np.all(hist[mode_i,1:]<=hist[mode_i,0]+1e-4)
            counts=np.array(ref['global_candidate_counts'])
            for policy_i,(_,scale) in enumerate(POLICIES):
                weight=np.ones(20) if scale=='raw' else bin_importance(counts,edges,scale)
                total[mode_i,policy_i]+=hist[mode_i]@weight
    rows_out=[]
    for m,mode in enumerate(MODES):
        for k,(policy,_) in enumerate(POLICIES):
            base=total[m,k,0];base_error=100*base[1]/base.sum() if base.sum() else None
            for fold in range(5):
                good,bad=total[m,k,fold+1];n=good+bad
                rows_out.append(dict(root_1718=str(cell),casey_coarse=row['casey_coarse'],mode=mode,policy=policy,fold=fold,
                    baseline_correct_weight=float(base[0]),baseline_incorrect_weight=float(base[1]),baseline_error_percent=base_error,
                    correct_weight=good,incorrect_weight=bad,error_percent=100*bad/n if n else None,
                    retained_weight_percent=100*n/base.sum() if base.sum() else None))
    savez(WORK/'histograms'/f'{cell}.npz',roots=np.array(row['historical_roots'],np.int64),histograms=np.array(all_hist),totals=total)
    write(path,dict(complete=True,root_1718=str(cell),seconds=time.monotonic()-started,rows=rows_out,
                   baseline_reproduced=True,plan_sha256=sha(WORK/'plan.json')))
    print(json.dumps(dict(cell=cell,seconds=time.monotonic()-started,roots=len(row['historical_roots']))),flush=True)


def worker():
    cells=read(WORK/'cells.json');failures=[]
    # Deterministic shuffle spreads filesystem probes among shared-pool workers.
    np.random.default_rng(int(os.environ.get('SLURM_ARRAY_TASK_ID',0))).shuffle(cells)
    for row in cells:
        cell=row['root_1718'];done=WORK/'cells'/f'{cell}.json'
        if done.exists():continue
        lock=WORK/'claims'/f'{cell}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if done.exists():continue
            try:score_cell(row)
            except Exception as exc:
                traceback.print_exc();failures.append(cell)
                write(WORK/'errors'/f'{cell}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    if failures:raise RuntimeError(f'{len(failures)} failed cells')


def summarize():
    import pandas as pd
    cells=read(WORK/'cells.json');rows_out=[];missing=[]
    for row in cells:
        path=WORK/'cells'/f"{row['root_1718']}.json"
        if not path.exists():missing.append(row['root_1718']);continue
        result=read(path);assert result['complete'] and result['baseline_reproduced'] and result['plan_sha256']==sha(WORK/'plan.json')
        rows_out.extend(result['rows'])
    if not rows_out:raise ValueError('No completed cells')
    df=pd.DataFrame(rows_out);df.to_csv(WORK/'cell_scores.csv',index=False)
    summaries=[]
    for label,frame in [('all_neurons',df)]+list(df.groupby('casey_coarse')):
        for (mode,policy,fold),part in frame.groupby(['mode','policy','fold']):
            bg=part.baseline_correct_weight.sum();bb=part.baseline_incorrect_weight.sum()
            g=part.correct_weight.sum();b=part.incorrect_weight.sum()
            summaries.append(dict(group=label,group_type='unresolved_casey' if label in {'PV','MC','AltDTC'} else ('cohort' if label=='all_neurons' else 'casey_family_or_thalamocortical'),mode=mode,policy=policy,fold=int(fold),neurons=len(part),
                baseline_correct_weight=bg,baseline_incorrect_weight=bb,correct_weight=g,incorrect_weight=b,
                baseline_error_percent=100*bb/(bg+bb) if bg+bb else None,
                error_percent=100*b/(g+b) if g+b else None,
                retained_weight_percent=100*(g+b)/(bg+bb) if bg+bb else None,
                incorrect_weight_removed_percent=100*(bb-b)/bb if bb else None,
                correct_weight_retained_percent=100*g/bg if bg else None,
                baseline_mean_neuron_error_percent=part.baseline_error_percent.mean(),
                baseline_median_neuron_error_percent=part.baseline_error_percent.median(),
                mean_neuron_error_percent=part.error_percent.mean(),median_neuron_error_percent=part.error_percent.median(),
                neurons_with_lower_error=int((part.error_percent<part.baseline_error_percent-1e-10).sum()),
                neurons_with_higher_error=int((part.error_percent>part.baseline_error_percent+1e-10).sum()),
                neurons_without_retained_pairs=int(part.error_percent.isna().sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(WORK/'pooled_scores.csv',index=False)
    summary[summary.group_type=='unresolved_casey'].to_csv(WORK/'unresolved_casey_comparison.csv',index=False)
    summary[summary.group_type=='casey_family_or_thalamocortical'].to_csv(WORK/'casey_family_comparison.csv',index=False)
    overall=summary[summary.group=='all_neurons'];overall.to_csv(WORK/'cohort_comparison.csv',index=False)
    write(WORK/'summary.json',dict(complete=not missing,completed_cells=len(cells)-len(missing),missing_cells=missing,
        plan=read(WORK/'plan.json'),cohort=json.loads(overall.to_json(orient='records'))))
    print(overall[overall.policy.isin(['all_pairs','uniform_distance'])].to_string(index=False))
    if missing:raise RuntimeError(f'{len(missing)} cells still missing')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','labels','worker','cell','summarize'])
    parser.add_argument('--rank',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)))
    parser.add_argument('--cell',type=int);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='labels':export_labels(args.rank)
    elif args.stage=='worker':worker()
    elif args.stage=='cell':score_cell(next(r for r in read(WORK/'cells.json') if int(r['root_1718'])==args.cell))
    else:summarize()
