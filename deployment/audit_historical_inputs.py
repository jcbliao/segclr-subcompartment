import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd().parent/'quantify_segclr_pairs/scripts'))
from historical_packed import RunStore
from segclr_db import store as st
work=Path('/orcd/scratch/orcd/013/jcbliao/segclr_pair_audit/historical117_embeddings_20261103');s=RunStore(work);p=json.loads((work/'plan.json').read_text())
ready={};published={}
for key,payload,_ in s.meta.values():
 if key.startswith('plans/') and key.endswith('/ready.json'):
  r=json.loads(payload);ready[int(r['root'])]=r
 if key.startswith('published/'):
  r=json.loads(payload);published[int(r['root'])]=r
assert set(ready)==set(p['roots'])==set(published)
for root in p['roots']:
 a,b=ready[root],published[root]
 assert b['exact_roundtrip'] and b['rows']==a['l2_nodes']+a['skeleton_vertices']
 assert b['checkpoint_sha256']==p['checkpoint_sha256'] and b['run_id']==p['run_id']
print('VERIFIED',len(ready),'roots',sum(r['rows'] for r in published.values()),'published nodes','preserved existing',sum(r.get('existing_vectors_preserved',0) for r in published.values()),flush=True)
store=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns');ds=st.open_table(store,'node_embeddings',64)
print('DATASET',ds.uri,'VERSION',ds.version,flush=True)
root=next(r for r in p['roots'] if ready[r]['l2_nodes']>1000)
clause=f"run_id='{p['run_id']}' AND checkpoint_id='{p['checkpoint_id']}' AND root_id={root} AND node_set_id IN ('chunkedgraph_l2_v117','cave_skeleton_v4')"
t=time.monotonic();data=ds.scanner(columns=['root_id','node_id','node_set_id','embedding'],filter=clause,batch_size=65536,batch_readahead=1,fragment_readahead=1).to_table()
print('LIVE SAMPLE',root,len(data),'expected',published[root]['rows'],'seconds',time.monotonic()-t,flush=True)
assert len(data)==published[root]['rows']
import pyarrow.parquet as pq
pq.write_table(data,'/tmp/historical_subcompartment_pilot.parquet')
Path('/tmp/historical_subcompartment_inventory.json').write_text(json.dumps(dict(source=p,ready=ready,published=published,snapshot=dict(uri=ds.uri,version=ds.version),pilot_root=root)))
