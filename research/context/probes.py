"""Five fixed contextual conditions x three seeds. VAL-only checkpoint selection."""
import argparse
import io
from pathlib import Path
import numpy as np
import feature_pipeline as f

CONDITIONS=('A','B','C','B_shuffled_local','C_shuffled_global')
SEEDS=(20261028,20261029,20261030)

def job_spec(job):
    f.need(type(job) is int and 0 <= job < 15,'Exactly fifteen fixed jobs')
    return CONDITIONS[job//3],SEEDS[job%3]

def inputs(x,donor_local,donor,condition):
    f.need(condition in CONDITIONS,'Unknown condition')
    z=x.copy()
    if condition=='A':z[:,:,5:]=0
    elif condition in ('B','B_shuffled_local'):z[:,:,37:]=0
    if condition=='B_shuffled_local':
        z[:,:,5:37]=donor_local[donor][:,np.arange(z.shape[1])%32,:]
    elif condition=='C_shuffled_global':z[:,:,37:]=x[donor,0,37:][:,None,:]
    f.need(np.array_equal(z[:,:,:5],x[:,:,:5]),'Center/coordinates changed')
    return z

def model():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(69,128),torch.nn.ReLU(),torch.nn.Linear(128,128),
                               torch.nn.ReLU(),torch.nn.Linear(128,3),torch.nn.Sigmoid())

def checked_completion(root,job):
    root=Path(root);condition,seed=job_spec(job)
    v=f.checked(f.read(root/'probes'/('%02d'%job)/'completion.json'))
    f.need(v['status']=='complete' and v['job']==job and v['condition']==condition and v['seed']==seed
           and v['request']==f.desc(root/'request.json') and v['test_accessed'] is False
           and v['features']==f.desc(root/'features/completion.json'),'Probe identity/request/selection binding')
    f.bound(v['checkpoint'])
    return v

def validation_loss(net,x,y,batch_size=4096):
    import torch
    total=0.0; count=0
    net.eval()
    with torch.no_grad():
        for offset in range(0,len(x),batch_size):
            loss=(net(x[offset:offset+batch_size])-y[offset:offset+batch_size]).square().mean(1)
            total+=float(loss.double().sum().item());count+=len(loss)
    return total/count

def train(root,job,device):
    import torch
    root=Path(root);f.request(root);f.torch_setup(device)
    condition,seed=job_spec(job);torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    completion=f.desc(root/'features/completion.json')
    tr=f.load_arrays(root,'train');va=f.load_arrays(root,'val')
    train_x=inputs(tr[0],tr[2],tr[3],condition).reshape(-1,69)
    val_x=inputs(va[0],va[2],va[3],condition).reshape(-1,69)
    tx=torch.as_tensor(train_x,device=device);ty=torch.as_tensor(tr[1].reshape(-1,3),device=device)
    vx=torch.as_tensor(val_x,device=device);vy=torch.as_tensor(va[1].reshape(-1,3),device=device)
    del tr,va,train_x,val_x
    net=model().to(device);optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    history=[];best=None;best_loss=float('inf');stale=0
    for epoch in range(1,21):
        net.train();order=np.random.default_rng(f.seed_for('batch',seed,epoch)).permutation(len(tx))
        indices=torch.as_tensor(order,device=device);total=0.;seen=0
        for offset in range(0,len(tx),4096):
            idx=indices[offset:offset+4096];optimizer.zero_grad(set_to_none=True)
            loss=(net(tx[idx])-ty[idx]).square().mean()
            f.need(torch.isfinite(loss).item(),'Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(idx);seen+=len(idx)
        vl=validation_loss(net,vx,vy)
        f.need(np.isfinite(vl),'Nonfinite VAL loss')
        history.append(dict(epoch=epoch,train_mse=total/seen,val_group_mean_mse=vl,
                            batch_order_sha256=f.hashlib.sha256(order.tobytes()).hexdigest()))
        print(condition,seed,history[-1],flush=True)
        if vl<best_loss:
            best_loss=vl;stale=0
            best=dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch=epoch,
                      val_group_mean_mse=vl,condition=condition,seed=seed,features=completion)
        else:stale+=1
        if stale>=5:break
    f.need(best is not None,'No selected checkpoint')
    out=root/'probes'/('%02d'%job);buf=io.BytesIO();torch.save(best,buf)
    f.publish(out/'best.pt',buf.getvalue())
    f.save(out/'completion.json',dict(status='complete',job=job,condition=condition,seed=seed,
           request=f.desc(root/'request.json'),features=completion,checkpoint=f.desc(out/'best.pt'),
           selected_epoch=best['epoch'],val_group_mean_mse=best_loss,history=history,
           architecture='69-128-ReLU-128-ReLU-3-Sigmoid',loss='MSE',optimizer='Adam_lr0.001',
           max_epochs=20,patience=5,batch_size=4096,test_accessed=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',type=int,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();train(a.root,a.job,a.device)

if __name__=='__main__':main()
