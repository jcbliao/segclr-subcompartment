import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pytest
pytest.importorskip("segclr_db")
sys.path.insert(0,str(Path(__file__).parent))
import historical as h
import historical_import_support as recovery
from segclr_db import store as st


def fixture_scores():
    n=7
    values={'root_id':pa.array([10]*4+[20]*3,pa.int64()),'node_id':pa.array([0,1,2,3,0,1,2],pa.int32())}
    raw=np.arange(n*6,dtype=np.float32).reshape(n,6)/7
    values['fold0_logits']=pa.FixedSizeListArray.from_arrays(pa.array(raw.ravel()),6)
    return pa.table(values)


def test_six_class_payload_and_identities():
    table=fixture_scores();pred,logits=h.expected_tables(table,0,'run','chunkedgraph_l2_v117',list('abcdef'))
    assert pred['pred_labels'].to_pylist()==[['f']]*7
    assert pred['synapse_id'].null_count==7
    assert pred['subject_root_id'].equals(pred['root_id'])
    assert all(v==0 for v in logits['level'].to_pylist())
    assert logits['logits'].to_pylist()==np.asarray(table['fold0_logits'].combine_chunks().values).reshape(7,6).tolist()
    for name,wanted in [('predictions',pred),('prediction_logits',logits)]:
        actual=wanted.take(pa.array([3,0,5]))
        missing=recovery.missing_rows(wanted,actual,name)
        assert len(missing)==4
        assert len(recovery.missing_rows(wanted,pa.concat_tables([actual,missing]),name))==0
        with pytest.raises(ValueError,match='Duplicate stored'):
            recovery.missing_rows(wanted,pa.concat_tables([actual,actual]),name)
        changed=wanted.set_column(wanted.schema.get_field_index('node_id'),'node_id',pa.array([99,1,2,3,0,1,2],pa.int32()))
        with pytest.raises(ValueError,match='Unexpected stored'):
            recovery.missing_rows(wanted,changed,name)


def test_real_lance_partial_append_recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'WORK',tmp_path/'work')
    h.save(recovery.WORK/'import_migration.json',dict(complete=True,plan_sha256='plan',fresh_scope_proof='isolated test',legacy_groups={}))
    store=st.init_store(tmp_path/'db','microns',datastack='minnie65_public',mat_version=1718)
    roots=np.array([10,20],np.int64)
    expected=h.expected_tables(fixture_scores(),0,'test_run','chunkedgraph_l2_v117',list('abcdef'))
    for name,wanted in zip(['predictions','prediction_logits'],expected):
        scope="prediction_run_id='test_run' AND node_set_id='chunkedgraph_l2_v117'"
        attempt=recovery._attempt_ledger(store,name,scope,roots,wanted,(0,'l2',0,'group'),'plan')
        st.append(store,name,wanted.slice(0,3))
        actual=recovery._scan_attempt(store,name,attempt,roots,wanted.schema)
        fresh=recovery.missing_rows(wanted,actual,name);assert len(fresh)==4
        st.append(store,name,fresh)
        after=recovery._scan_attempt(store,name,attempt,roots,wanted.schema)
        assert len(after)==7 and len(recovery.missing_rows(wanted,after,name))==0


def test_production_size_label_array():
    # NumPy Unicode -> Arrow starts chunking at this production scale.
    n=2_450_000
    raw=np.zeros((n,6),np.float32)
    raw[np.arange(n),np.arange(n)%6]=1
    table=pa.table({'root_id':np.ones(n,np.int64),'node_id':np.arange(n,dtype=np.int32),
        'fold0_logits':pa.FixedSizeListArray.from_arrays(pa.array(raw.ravel()),6)})
    classes=['astrocytic_process','axon','dendrite','microglial_process','oligodendrocytic_process','soma']
    pred,logits=h.expected_tables(table,0,'run','chunkedgraph_l2_v117',classes)
    assert len(pred)==len(logits)==n
    assert pred['pred_labels'][n-1].as_py()==[classes[(n-1)%6]]
