"""Six-class copy of the established ResNet trainer; four-class trainer is frozen."""
import argparse
import json
import time
import sys
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report,log_loss
from six_class_data import BASE,CLASSES,load,weights,write_json

class LinearProbe(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.linear=nn.Linear(64,len(CLASSES))
    def forward(self,x):
        return self.linear((x-self.mean)/self.scale)


class ResNetProbe(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent.parent/'gnn_classifier'))
        from gnn.resnet import DeepResNetTrunk
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.trunk=DeepResNetTrunk(64,hidden_size=128,hidden_layers=4,dropout_rate=0.0,use_bn=False)
        self.output=nn.Linear(128,len(CLASSES))
    def forward(self,x):
        return self.output(self.trunk((x-self.mean)/self.scale))


def make_model(head,mean,scale):
    return (LinearProbe if head=='linear' else ResNetProbe)(mean,scale)


def evaluate_epoch(model,x,test_ids,truth,sample_weights,epoch):
    model.eval()
    predictions=[]
    device=next(model.parameters()).device
    with torch.inference_mode():
        for start in tqdm(range(0,len(test_ids),65536),desc=f'Epoch {epoch} eval',leave=False,mininterval=5,file=sys.stderr):
            features=torch.from_numpy(x[test_ids[start:start+65536]]).to(device)
            predictions.append(model(features).argmax(dim=1).cpu().numpy())
    pred=np.concatenate(predictions)
    per_class=f1_score(truth,pred,labels=np.arange(len(CLASSES)),average=None,zero_division=0)
    return dict(test_macro_f1=float(per_class.mean()),
                test_f1_per_class=dict(zip(CLASSES,map(float,per_class))),
                test_cell_class_balanced_macro_f1=float(f1_score(truth,pred,labels=np.arange(len(CLASSES)),average='macro',sample_weight=sample_weights,zero_division=0)))


def balanced_batches(labels,sampling_weights,batch_size,steps):
    """Exactly equal classes per batch; equal expected cell weight within class."""
    if batch_size%len(CLASSES):raise ValueError('Batch size must be divisible by class count')
    per_class=batch_size//len(CLASSES)
    draws=[]
    for cls in range(len(CLASSES)):
        indices=torch.nonzero(labels==cls,as_tuple=False).flatten()
        if not len(indices):raise ValueError(f'Missing training class {cls}')
        selected=torch.multinomial(sampling_weights[indices],steps*per_class,replacement=True)
        draws.append(indices[selected].reshape(steps,per_class))
    for step in range(steps):
        yield torch.cat([samples[step] for samples in draws])




def inverse_sqrt_batches(labels,batch_size,steps):
    counts=torch.bincount(labels,minlength=len(CLASSES)).float()
    if torch.any(counts==0):raise ValueError('Missing training class')
    sample_weights=counts.rsqrt()[labels]
    draws=torch.multinomial(sample_weights,steps*batch_size,replacement=True)
    for start in range(0,len(draws),batch_size):
        yield draws[start:start+batch_size]


def equal_cell_type_weights(labels,roots,root_types):
    """Sqrt compartment frequency; uniform type, uniform cell, uniform point."""
    counts=np.bincount(labels,minlength=len(CLASSES))
    if np.any(counts==0):raise ValueError('Missing class')
    class_probability=np.sqrt(counts);class_probability/=class_probability.sum()
    result=np.empty(len(labels),np.float32)
    for cls in range(len(CLASSES)):
        ids=np.flatnonzero(labels==cls)
        cells,points_per_cell=np.unique(roots[ids],return_counts=True)
        types=np.asarray([root_types[int(cell)] for cell in cells])
        type_names,cells_per_type=np.unique(types,return_counts=True)
        type_index=np.searchsorted(type_names,types)
        probability_per_cell=class_probability[cls]/(len(type_names)*cells_per_type[type_index])
        cell_positions=np.searchsorted(cells,roots[ids])
        result[ids]=(probability_per_cell/points_per_cell)[cell_positions]
    return result


def probability_batches(probabilities,batch_size,steps):
    draws=torch.multinomial(probabilities,steps*batch_size,replacement=True)
    for start in range(0,len(draws),batch_size):
        yield draws[start:start+batch_size]


def run(variant,epochs=20,head="linear",sampling="balanced"):
    lr=1e-4 if head=="resnet" else 0.01
    weight_decay=1e-5 if head=="resnet" else 1e-4
    if not torch.cuda.is_available():raise RuntimeError('GPU training requires CUDA')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False
    folder=('models_gpu_class_balanced' if epochs==20 else f'models_gpu_class_balanced_{epochs}epochs') if head=='linear' else f'models_resnet_class_balanced_{epochs}epochs'
    if sampling!='balanced':folder=f'models_{head}_{sampling}_{epochs}epochs'
    out=BASE/folder/variant;out.mkdir(parents=True,exist_ok=True)
    x,y,roots,train,root_ids=load(variant)
    train_ids=np.flatnonzero(train);test_ids=np.flatnonzero(~train)
    assert not set(roots[train_ids]) & set(roots[test_ids])
    w=weights(y[train_ids],roots[train_ids]);eval_w=weights(y[test_ids],roots[test_ids])
    stats=dict(n_train=len(train_ids),n_test=len(test_ids),n_train_cells=len(np.unique(roots[train_ids])),n_test_cells=len(np.unique(roots[test_ids])),train_counts={name:int(np.sum(y[train_ids]==i)) for i,name in enumerate(CLASSES)},test_counts={name:int(np.sum(y[test_ids]==i)) for i,name in enumerate(CLASSES)})
    print('DATA',variant,stats,flush=True)
    scaler=StandardScaler()
    for start in tqdm(range(0,len(train_ids),65536),desc="Fit training scaler",mininterval=5,file=sys.stderr):
        scaler.partial_fit(x[train_ids[start:start+65536]],sample_weight=w[start:start+65536])
    model=make_model(head,scaler.mean_,scaler.scale_).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    training_x=torch.from_numpy(np.ascontiguousarray(x[train_ids]))
    training_y=torch.from_numpy(y[train_ids].astype(np.int64))
    training_w=torch.from_numpy(w)
    type_weights=None
    if sampling=='equal_cell_types':
        source=json.loads((__import__('pathlib').Path(__file__).resolve().parent.parent/'gnn_classifier/data/manifest.json').read_text())['cells']
        root_types=[source[root]['cell_type'] for root in root_ids]
        type_weights=torch.from_numpy(equal_cell_type_weights(y[train_ids],roots[train_ids],root_types))
    free,total=torch.cuda.mem_get_info()
    needed=training_x.numel()*4+training_y.numel()*8+training_w.numel()*4+len(train_ids)*8
    resident=needed<0.6*free
    if resident:
        training_x=training_x.cuda();training_y=training_y.cuda();training_w=training_w.cuda()
        if type_weights is not None:type_weights=type_weights.cuda()
    expected=np.sqrt(np.bincount(y[train_ids],minlength=len(CLASSES))) if sampling!='balanced' else np.ones(len(CLASSES))
    expected=expected/expected.sum()
    print('SAMPLING',sampling,'expected_class_fractions',dict(zip(CLASSES,map(float,expected))),flush=True)
    print('HEAD',head,'parameters',sum(p.numel() for p in model.parameters()),'lr',lr,'weight_decay',weight_decay,flush=True)
    print('GPU',torch.cuda.get_device_name(),'resident_data',resident,'dataset_bytes',needed,flush=True)
    batch_size=16384;history=[];best_f1=-1.;best_epoch=None
    epoch_bar=tqdm(range(epochs),desc=f"{variant} epochs",mininterval=1,file=sys.stderr)
    for epoch in epoch_bar:
        begin=time.monotonic();model.train()
        steps=(len(train_ids)+batch_size-1)//batch_size
        if sampling=='balanced':sampler=balanced_batches(training_y,training_w,batch_size,steps)
        elif sampling=='inverse_sqrt':sampler=inverse_sqrt_batches(training_y,batch_size,steps)
        else:sampler=probability_batches(type_weights,batch_size,steps)
        total_loss=torch.zeros((),device='cuda')
        for ids in tqdm(sampler,total=steps,desc=f"Epoch {epoch+1} train",leave=False,mininterval=5,file=sys.stderr):
            features=training_x[ids].to('cuda');labels=training_y[ids].to('cuda')
            optimizer.zero_grad(set_to_none=True)
            losses=nn.functional.cross_entropy(model(features),labels,reduction='none')
            loss=losses.mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step();total_loss+=loss.detach()*len(ids)
        torch.cuda.synchronize()
        metric=dict(epoch=epoch+1,train_loss=float(total_loss/(steps*batch_size)),seconds=time.monotonic()-begin)
        metric.update(evaluate_epoch(model,x,test_ids,y[test_ids],eval_w,epoch+1))
        if metric['test_macro_f1']>best_f1:
            best_f1=metric['test_macro_f1'];best_epoch=epoch+1
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,classes=CLASSES,embedding_dim=64,variant=variant,head=head,sampling=sampling,best_epoch=best_epoch,best_macro_f1=best_f1,selection_split='full existing manifest test split'),out/'model.pt')
        metric.update(best_epoch=best_epoch,best_macro_f1=best_f1)
        history.append(metric)
        epoch_bar.set_postfix(loss=metric['train_loss'],f1=metric['test_macro_f1'],best=best_f1)
        print(f"EPOCH {epoch+1}/{epochs} loss={metric['train_loss']:.6f} test_macro_f1={metric['test_macro_f1']:.6f} per_class_f1={metric['test_f1_per_class']} best_epoch={best_epoch} best_f1={best_f1:.6f}",flush=True)
        # Checkpoint selection follows the user's full-test-split F1 rule.
        write_json(out/'history.json',history)
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    state=checkpoint['state_dict']
    model.load_state_dict(state);model.eval()
    if head=='linear':
        effective_weight=state['linear.weight'].numpy()/state['scale'].numpy()[None,:]
        effective_bias=state['linear.bias'].numpy()-effective_weight@state['mean'].numpy()
        np.savez(out/'linear_coefficients.npz',weight=effective_weight,bias=effective_bias,classes=CLASSES)
    with torch.inference_mode():
        restored=make_model(head,scaler.mean_,scaler.scale_).cuda()
        restored.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
        check=torch.from_numpy(x[test_ids[:128]]).cuda()
        torch.testing.assert_close(model(check),restored(check),rtol=0,atol=0)
        probabilities=[]
        for start in tqdm(range(0,len(test_ids),65536),desc="Evaluate best checkpoint",mininterval=5,file=sys.stderr):
            features=torch.from_numpy(x[test_ids[start:start+65536]]).cuda()
            probabilities.append(torch.softmax(model(features),dim=1).cpu().numpy())
    probs=np.concatenate(probabilities);pred=probs.argmax(axis=1);truth=y[test_ids]
    metrics=dict(accuracy=float(accuracy_score(truth,pred)),balanced_accuracy=float(balanced_accuracy_score(truth,pred)),macro_f1=float(f1_score(truth,pred,average='macro')),cell_class_balanced_accuracy=float(accuracy_score(truth,pred,sample_weight=eval_w)),cell_class_balanced_macro_f1=float(f1_score(truth,pred,average='macro',sample_weight=eval_w)),log_loss=float(log_loss(truth,probs,labels=np.arange(len(CLASSES)))),confusion_matrix=confusion_matrix(truth,pred,labels=np.arange(len(CLASSES))).tolist(),classification_report=classification_report(truth,pred,labels=np.arange(len(CLASSES)),target_names=CLASSES,output_dict=True,zero_division=0))
    assert abs(metrics['macro_f1']-best_f1)<1e-12
    metrics.update(best_epoch=best_epoch,selection_split='full existing 20% manifest test split',selection_metric='macro_f1')
    write_json(out/'metrics.json',metrics)
    write_json(out/'training.json',dict(variant=variant,classes=CLASSES,head=head,model=(f'single nn.Linear(64,{len(CLASSES)})' if head=='linear' else f'gnn.resnet.DeepResNetTrunk(64,128,4) + nn.Linear(128,{len(CLASSES)}); dropout=0, BN=False'),optimizer='AdamW',learning_rate=lr,weight_decay=weight_decay,epochs=epochs,batch_size=batch_size,random_seed=0,standardization='weighted training samples only',sampling=sampling,sampling_details=('point weights 1/sqrt(training class count), with replacement; uniform points within each class' if sampling=='inverse_sqrt' else 'sqrt compartment frequency; uniform cell type within compartment, uniform cell within type, uniform point within cell; replacement' if sampling=='equal_cell_types' else 'equal samples per class per batch; equal expected cell frequency within class; replacement'),expected_class_fractions=dict(zip(CLASSES,map(float,expected))),loss_weighting='none; balancing is implemented by sampling',samples_per_epoch=steps*batch_size,best_epoch=best_epoch,selection_metric='macro_f1',test_policy='full existing 20% manifest test split evaluated every epoch; best checkpoint selected on this split',gpu=torch.cuda.get_device_name(),gpu_resident_data=resident,**stats))
    np.savez_compressed(out/'test_predictions.npz',y_true=truth,y_pred=pred,probabilities=probs,root_ids=np.asarray(root_ids)[roots[test_ids]])
    print('DONE',variant,metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['geometric','geodesic'],required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--head',choices=['linear','resnet'],default='linear')
    p.add_argument('--sampling',choices=['balanced','inverse_sqrt','equal_cell_types'],default='balanced')
    args=p.parse_args()
    if args.epochs<1:p.error('--epochs must be positive')
    assert (BASE/'manifest.json').exists();run(args.variant,args.epochs,args.head,args.sampling)
