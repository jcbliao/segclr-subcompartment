"""Train and evaluate a linear logistic probe for soma/process embeddings."""
import argparse
from pathlib import Path
import json
import time
import sys
import numpy as np
import joblib
from tqdm.auto import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report,log_loss
from prepare_compartment_training import BASE,CLASSES,write_json


def load(variant):
    xs=[];ys=[];roots=[];splits=[];root_ids=[]
    for folder in [BASE/'soma_cells',BASE/variant/'cells']:
        for path in tqdm(sorted(folder.glob('*.npz')),desc=f'Load {folder.parent.name}/{folder.name}',mininterval=5,file=sys.stderr):
            a=np.load(path);x=a['x'];y=a['y'];root=str(a['root_id']);split=str(a['split'])
            if not len(y):continue
            if root not in root_ids:root_ids.append(root)
            root_index=root_ids.index(root)
            assert x.shape==(len(y),64) and np.isfinite(x).all()
            xs.append(x);ys.append(y);roots.append(np.full(len(y),root_index,np.int32));splits.append(np.full(len(y),split=='train',bool))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(roots),np.concatenate(splits),root_ids


def weights(y,roots):
    # Equal total weight per class, then equal weight per cell within each class.
    w=np.empty(len(y),np.float32)
    for cls in range(4):
        ids=np.flatnonzero(y==cls)
        cell_ids,counts=np.unique(roots[ids],return_counts=True)
        if not len(ids):raise ValueError(f'Missing class {CLASSES[cls]}')
        count_map=dict(zip(cell_ids,counts))
        w[ids]=[len(y)/(4*len(cell_ids)*count_map[r]) for r in roots[ids]]
    return w


def run(variant):
    out=BASE/'models'/variant;out.mkdir(parents=True,exist_ok=True)
    x,y,roots,train,root_ids=load(variant)
    train_ids=np.flatnonzero(train);test_ids=np.flatnonzero(~train)
    assert not set(roots[train_ids]) & set(roots[test_ids])
    w=weights(y[train_ids],roots[train_ids]);eval_w=weights(y[test_ids],roots[test_ids])
    stats=dict(n_train=len(train_ids),n_test=len(test_ids),n_train_cells=len(np.unique(roots[train_ids])),n_test_cells=len(np.unique(roots[test_ids])),train_counts={name:int(np.sum(y[train_ids]==i)) for i,name in enumerate(CLASSES)},test_counts={name:int(np.sum(y[test_ids]==i)) for i,name in enumerate(CLASSES)})
    print('DATA',variant,stats,flush=True)
    scaler=StandardScaler()
    batch=65536
    for start in range(0,len(train_ids),batch):
        ids=train_ids[start:start+batch]
        scaler.partial_fit(x[ids],sample_weight=w[start:start+batch])
    clf=SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,learning_rate='optimal',average=True,random_state=0)
    rng=np.random.default_rng(0)
    for epoch in range(20):
        begin=time.monotonic();order=rng.permutation(len(train_ids))
        for start in range(0,len(order),4096):
            pos=order[start:start+4096];ids=train_ids[pos]
            clf.partial_fit(scaler.transform(x[ids]),y[ids],classes=np.arange(4),sample_weight=w[pos])
        print(f'EPOCH {epoch+1}/20 {variant} seconds={time.monotonic()-begin:.1f}',flush=True)
    pipeline=Pipeline([('scaler',scaler),('classifier',clf)])
    joblib.dump(pipeline,out/'model.joblib')
    pred=[];probs=[]
    for start in range(0,len(test_ids),batch):
        ids=test_ids[start:start+batch];pred.append(pipeline.predict(x[ids]));probs.append(pipeline.predict_proba(x[ids]))
    pred=np.concatenate(pred);probs=np.concatenate(probs);truth=y[test_ids]
    metrics=dict(accuracy=float(accuracy_score(truth,pred)),balanced_accuracy=float(balanced_accuracy_score(truth,pred)),macro_f1=float(f1_score(truth,pred,average='macro')),cell_class_balanced_accuracy=float(accuracy_score(truth,pred,sample_weight=eval_w)),cell_class_balanced_macro_f1=float(f1_score(truth,pred,average='macro',sample_weight=eval_w)),log_loss=float(log_loss(truth,probs,labels=np.arange(4))),confusion_matrix=confusion_matrix(truth,pred,labels=np.arange(4)).tolist(),classification_report=classification_report(truth,pred,labels=np.arange(4),target_names=CLASSES,output_dict=True,zero_division=0))
    write_json(out/'metrics.json',metrics)
    write_json(out/'training.json',dict(variant=variant,classes=CLASSES,model='SGDClassifier linear one-vs-rest logistic regression; no hidden layers',epochs=20,alpha=1e-4,random_seed=0,batch_size=4096,standardization='weighted training samples only',weighting='equal class weight; equal cell weight within each class',hyperparameter_selection='fixed before held-out test evaluation',test_policy='existing manifest test split evaluated once after training',**stats))
    np.savez_compressed(out/'test_predictions.npz',y_true=truth,y_pred=pred,probabilities=probs,root_ids=np.asarray(root_ids)[roots[test_ids]])
    np.testing.assert_array_equal(joblib.load(out/'model.joblib').predict(x[test_ids[:128]]),pred[:128])
    print('DONE',variant,metrics,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['geometric','geodesic'],required=True)
    args=parser.parse_args();assert (BASE/'manifest.json').exists();run(args.variant)

if __name__=='__main__':main()
