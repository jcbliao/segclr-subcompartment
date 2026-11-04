"""Prepare soma/process linear-classifier data with verified vertex correspondence."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import os
import sys
import struct
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
from scipy.spatial import cKDTree
import torch
from tqdm.auto import tqdm
from estimate_microns_soma import HERE,geometry

BASE=Path(os.environ.get('SUBCOMPARTMENT_DATA_DIR','/orcd/scratch/orcd/013/jcbliao/segclr/subcompartment_linear_v1'))
GNN=HERE.parent/'gnn_classifier'
LABELS=Path('/orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/ossify_compartments/all_cells/skeletons')
SOMA=Path('/orcd/scratch/orcd/013/jcbliao/segclr/soma_center_embeddings_v1718')
CLASSES=['soma','dendrite','axon','astrocytic_process']
EXCLUDE={'864691136364187874'}


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def near_geodesic(graph,source,n):
    if not len(source):return np.full(n,np.inf)
    return dijkstra(graph,directed=False,indices=source,min_only=True,limit=5000.)


def mark_soma(rid,v,labels):
    dest=HERE/'soma_dataset'/rid
    surface=np.load(dest/'surface.npz');p=json.loads((dest/'provenance.json').read_text())
    center=np.array(p['center_nm'])
    syn=pd.read_csv(dest/'synapses_nm.csv')
    xyz=syn.loc[syn.compartment=='soma',list(geometry.POSITION_COLS)].to_numpy()
    r=np.linalg.norm(xyz-center,axis=1);xyz=xyz[r<=2.5*np.median(r)]
    radii,dirs=geometry.to_directions(xyz,center)
    d=v-center;dist=np.linalg.norm(d,axis=1)
    ids=np.flatnonzero(dist<=radii.max())
    for start in range(0,len(ids),512):
        sub=ids[start:start+512];valid=dist[sub]>0
        inside=~valid
        if valid.any():
            chosen=sub[valid]
            radius=geometry.kernel_radius_at(d[chosen]/dist[chosen,None],dirs,radii,**{k:p['parameters'][k] for k in ['bandwidth_deg','radial_quantile','offset_nm']})[0]
            inside[valid]=dist[chosen]<radius
        labels[sub[inside]]=1
    return labels


def one(rid,meta,center):
    paths={variant:BASE/variant/'cells'/f'{rid}.npz' for variant in ['geometric','geodesic']}
    audit=BASE/'audit'/f'{rid}.json'
    if audit.exists() and all(p.exists() for p in paths.values()):return json.loads(audit.read_text())
    payload=(LABELS/rid).read_bytes();n,e=struct.unpack_from('<II',payload)
    v=np.frombuffer(payload,'<f4',3*n,8).reshape(n,3).astype(np.float64)
    edges=np.frombuffer(payload,'<u4',2*e,8+12*n).reshape(e,2).astype(np.int64)
    labels=np.frombuffer(payload,'<f4',n,8+12*n+8*e+4*n).astype(np.int8).copy()
    graph=torch.load(GNN/'data/graph_cache'/f'{rid}.pt',weights_only=False)
    ids=graph.orig_node_ids.numpy();x=graph.x.numpy().astype(np.float32)
    if n!=graph.n_nodes_skeleton or not np.array_equal(v[ids].astype(np.float32),graph.pos.numpy()):
        raise ValueError(f'{rid}: embedding/skeleton vertex correspondence failed')
    if meta['cell_type']=='astrocyte':labels[:]=4
    elif meta['cell_type']=='thalamocortical':labels[:]=2
    else:labels=mark_soma(rid,v,labels)
    src,dst=edges.T
    length=np.linalg.norm(v[src]-v[dst],axis=1)
    adjacency=coo_matrix((np.r_[length,length],(np.r_[src,dst],np.r_[dst,src])),shape=(n,n)).tocsr()
    eligible=np.ones(n,dtype=bool)
    axon=np.flatnonzero(labels==2)
    if len(axon):
        _,components=connected_components(adjacency[axon][:,axon],directed=False)
        largest=np.argmax(np.bincount(components))
        eligible[axon[components!=largest]]=False
    cross=labels[src]!=labels[dst]
    eligible[np.unique(edges[cross])]=False
    eligible &= np.linalg.norm(v-np.asarray(center),axis=1)>10000.
    eligible &= labels!=1
    record=dict(root_id=rid,cell_type=meta['cell_type'],split=meta['split'],n_embedded=len(ids),vertex_mapping='exact float32 xyz and original index',n_class_boundary_edges=int(cross.sum()))
    for variant in paths:
        keep=eligible.copy()
        for own,other in [(2,3),(3,2)]:
            query=np.flatnonzero(labels==own);reference=np.flatnonzero(labels==other)
            if not len(query) or not len(reference):continue
            if variant=='geometric':distance=cKDTree(v[reference]).query(v[query])[0]
            else:distance=near_geodesic(adjacency,reference,n)[query]
            keep[query[distance<=5000.]]=False
        selected=keep[ids]
        target=np.array([{3:1,2:2,4:3}.get(int(label),-1) for label in labels[ids]],dtype=np.int8)
        if np.any(target[selected]<0):raise ValueError('Unexpected class')
        np.savez_compressed(paths[variant],x=x[selected],y=target[selected],point_id=ids[selected],xyz_nm=v[ids[selected]],root_id=rid,split=meta['split'])
        record[variant]={CLASSES[i]:int(np.sum(target[selected]==i)) for i in range(4)}
    write_json(audit,record)
    return record


def prepare_soma(manifest):
    for part in sorted((SOMA/'embeddings').glob('part-*.parquet')):
        frame=pd.read_parquet(part)
        for rid,group in frame.groupby('root_id'):
            rid=str(rid)
            if rid in EXCLUDE:continue
            dest=BASE/'soma_cells'/f'{rid}.npz'
            # Input parquet shards can split a cell; aggregate all pieces below.
            folder=BASE/'soma_parts';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{rid}-{part.stem}.npz',x=np.stack(group.embedding).astype(np.float32),point_id=group.point_id.to_numpy(),xyz_nm=group[['x_nm','y_nm','z_nm']].to_numpy())
    for rid in manifest['cells']:
        parts=list((BASE/'soma_parts').glob(f'{rid}-*.npz'))
        if not parts:continue
        arrays=[np.load(p) for p in parts];x=np.concatenate([a['x'] for a in arrays]);point_ids=np.concatenate([a['point_id'] for a in arrays]);coords=np.concatenate([a['xyz_nm'] for a in arrays])
        assert len(np.unique(point_ids))==len(point_ids)
        np.savez_compressed(BASE/'soma_cells'/f'{rid}.npz',x=x,y=np.zeros(len(x),np.int8),point_id=point_ids,xyz_nm=coords,root_id=rid,split=manifest['cells'][rid]['split'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--pilot',action='store_true');args=p.parse_args()
    manifest=json.loads((GNN/'data/manifest.json').read_text());centers=json.loads((GNN/'data/nucleus_positions.json').read_text())['positions']
    for folder in ['audit','soma_cells','geometric/cells','geodesic/cells']:(BASE/folder).mkdir(parents=True,exist_ok=True)
    eligible={r:m for r,m in manifest['cells'].items() if m['cell_type'] not in {'microglia','oligo','OPC'} and r not in EXCLUDE}
    if args.pilot:eligible={r:eligible[r] for r in ['864691134919403530',next(r for r,m in eligible.items() if m['cell_type']=='astrocyte'),next(r for r,m in eligible.items() if m['cell_type']=='thalamocortical')]}
    records=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,r,m,centers[r]):r for r,m in eligible.items()}
        for future in tqdm(as_completed(futures),total=len(futures),desc="Prepare cell labels",mininterval=5,file=sys.stderr):
            r=futures[future]
            try:record=future.result();records.append(record);print(f'OK {len(records)}/{len(eligible)} {r} {record["geometric"]}',flush=True)
            except Exception as error:failures.append(dict(root_id=r,error=str(error)));print('FAILED',r,error,flush=True)
    write_json(BASE/('pilot_failures.json' if args.pilot else 'preparation_failures.json'),failures)
    if failures:raise SystemExit('Preparation failed; training blocked until resolved')
    if args.pilot:return
    prepare_soma(manifest)
    write_json(BASE/'manifest.json',dict(classes=CLASSES,source_manifest=str(GNN/'data/manifest.json'),eligible_process_cells=len(eligible),excluded_cell=sorted(EXCLUDE),excluded_cell_types=['microglia','oligo','OPC'],selection='all applicable manifest cells',split_policy='existing cell-level train/test; no point-level random split',variants=['geometric','geodesic'],rules=json.loads((HERE/'curation/dataset_variants.json').read_text()),soma_embeddings=str(SOMA),embedding_dim=64))
    print('PREPARATION COMPLETE',flush=True)

if __name__=='__main__':main()
