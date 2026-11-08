"""Snapshot of the established exact-write/readback and retry-recovery helpers."""
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from historical import WORK, read, save, sha, st
SCAN_OPTIONS=dict(use_scalar_index=False,batch_size=65536,batch_readahead=1,fragment_readahead=1,io_buffer_size=64*2**20)

def _keys(table,name):
    fields=[('root_id','<i8'),('node_id','<i4')]
    if name=='prediction_logits':fields.append(('level','i1'))
    result=np.empty(len(table),dtype=np.dtype(fields))
    for key,_ in fields:result[key]=np.asarray(table[key])
    return result

def missing_rows(expected,actual,name):
    """Return absent keys while refusing duplicate keys or differing values."""
    wanted_keys=_keys(expected,name);want_order=np.argsort(wanted_keys)
    keys=wanted_keys[want_order]
    if len(keys)>1 and np.any(keys[1:]==keys[:-1]):
        raise ValueError(f'Duplicate expected {name} identities')
    if not len(actual):return expected
    got_keys=_keys(actual,name);got_order=np.argsort(got_keys);got_keys=got_keys[got_order]
    if len(got_keys)>1 and np.any(got_keys[1:]==got_keys[:-1]):
        raise ValueError(f'Duplicate stored {name} identities; refusing to overwrite')
    pos=np.searchsorted(keys,got_keys)
    if np.any(pos>=len(keys)) or not np.array_equal(keys[pos],got_keys):
        raise ValueError(f'Unexpected stored {name} keys in this node scope')
    want=expected.take(pa.array(want_order[pos],pa.int64()))
    got=actual.take(pa.array(got_order,pa.int64())).cast(expected.schema)
    if not want.equals(got):
        raise ValueError(f'Conflicting stored {name} values; refusing to overwrite')
    absent=np.ones(len(expected),bool);absent[want_order[pos]]=False
    return expected.filter(pa.array(absent))

def _scan_dataset_roots(dataset,scope,roots,schema,fragments=None):
    """Read only explicitly owned append fragments, with bounded filtering."""
    if fragments is not None and not fragments:
        return pa.Table.from_batches([],schema=schema)
    values=pa.array(roots,pa.int64());parts=[]
    reader=dataset.scanner(columns=schema.names,filter=scope,
                           fragments=fragments,**SCAN_OPTIONS).to_reader()
    for batch in reader:
        table=pa.Table.from_batches([batch])
        table=table.filter(pc.is_in(table['root_id'],value_set=values))
        if len(table):parts.append(table.cast(schema))
    return pa.concat_tables(parts) if parts else pa.Table.from_batches([],schema=schema)

def _attempt_ledger(store,name,scope,roots,wanted,group,plan_sha):
    """Persist fragment ownership before an append can create any fact rows."""
    fold,kind,rank,digest=group
    path=WORK/'attempts'/f'fold{fold}'/kind/f'{rank:02d}'/(name+'.json')
    identity=dict(plan_sha256=plan_sha,group_sha256=digest,table=name,scope=scope,
                  roots_sha256=hashlib.sha256(np.asarray(roots,np.int64).tobytes()).hexdigest())
    if path.exists():
        attempt=read(path)
        if any(attempt.get(k)!=v for k,v in identity.items()):
            raise ValueError('Append-attempt ledger belongs to different immutable inputs')
        return attempt
    migration=read(WORK/'import_migration.json')
    if (migration.get('plan_sha256')!=plan_sha or migration.get('complete') is not True or
            not migration.get('fresh_scope_proof')):
        raise ValueError('Missing ownership proof for direct append; inspect prior publishers')
    legacy=migration.get('legacy_fallback',migration.get('legacy_groups',{})).get(str(fold),[])
    is_legacy=any(x['kind']==kind and int(x['rank'])==rank for x in legacy)
    dataset=st.open_table(store,name) if st.table_exists(store,name) else None
    base_ids=[int(fragment.fragment_id) for fragment in dataset.get_fragments()] if dataset is not None else []
    attempt=dict(**identity,mode='legacy-baseline' if is_legacy else 'fresh-owned-run',
                 base_version=int(dataset.version) if dataset is not None else None,
                 base_fragment_ids=base_ids,
                 ownership_proof=str(WORK/'import_migration.json'))
    if is_legacy:
        # Only groups possibly touched by the previous publisher need this
        # initial whole-table reconciliation. Cache its exact rows durably;
        # every subsequent append/retry follows the bounded fragment ledger.
        baseline=(_scan_dataset_roots(dataset,scope,roots,wanted.schema) if dataset is not None
                  else wanted.slice(0,0))
        missing_rows(wanted,baseline,name)
        dest=path.with_suffix('.baseline.parquet');dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix('.tmp.parquet');pq.write_table(baseline,temp,compression='zstd');temp.replace(dest)
        attempt.update(baseline_path=str(dest),baseline_sha256=sha(dest),baseline_rows=len(baseline))
    # Atomic write-ahead: st.append is forbidden until this has committed.
    save(path,attempt)
    return attempt

def _scan_attempt(store,name,attempt,roots,schema):
    if not st.table_exists(store,name):return pa.Table.from_batches([],schema=schema)
    dataset=st.open_table(store,name);fragments=dataset.get_fragments()
    base=set(attempt['base_fragment_ids'])
    current={int(fragment.fragment_id) for fragment in fragments}
    if not base.issubset(current):
        # Compaction or a deletion rewrote a baseline fragment. Its cached
        # rows no longer describe the current physical snapshot, so reconcile
        # this one legacy group directly instead of double-counting them.
        return _scan_dataset_roots(dataset,attempt['scope'],roots,schema)
    added=[fragment for fragment in fragments if int(fragment.fragment_id) not in base]
    new=_scan_dataset_roots(dataset,attempt['scope'],roots,schema,added)
    if attempt['mode']=='legacy-baseline':
        path=Path(attempt['baseline_path'])
        if sha(path)!=attempt['baseline_sha256']:raise ValueError('Legacy recovery baseline changed')
        baseline=pq.read_table(path).cast(schema)
        return pa.concat_tables([baseline,new])
    return new
