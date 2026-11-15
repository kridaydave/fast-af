import sys,time,json,gc
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,'/content/d1-opt/src')
from scripts import d1_model as dm
from scripts.train_diffusion import TrainConfig,ShardLoader,block_diffusion_mask,build_optimizer,optimizer_step,model_masked_loss
r=Path('/content/d1-opt');torch.set_num_threads(1)
assert torch.cuda.get_device_capability()==(7,5)
result={'variants':[]}
for backend,fused,graph in [('chunk',False,False),('chunk',True,False),('chunk-tc',True,False),('chunk-tc',True,True),('chunk',False,False)]:
 torch.manual_seed(73);torch.cuda.manual_seed_all(73)
 m=dm.D1LM(32000,1024,backend=backend).cuda().train()
 cfg=TrainConfig(bundle=str(r/'src/probe-bundle'),out='unused',profile='d1',gdn_backend=backend,amp=True,device='cuda',fused_optimizer=fused)
 opt=build_optimizer(m,cfg);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
 scaler=torch.amp.GradScaler('cuda',init_scale=8)
 targets=next(ShardLoader(cfg.bundle,batch_size=4,seed=17,vocab=32000)).cuda();inputs,mask=block_diffusion_mask(targets,32,frontier=31)
 active=torch.zeros_like(mask);active[:,-32:]=True
 def forward_backward():
  with torch.autocast('cuda',dtype=torch.float16):
   if graph:
    logits=m(inputs,logits_last=32)
    loss=(F.cross_entropy(logits.flatten(0,1),targets[:,-32:].flatten(),reduction='none')*mask[:,-32:].flatten()).sum()/mask.sum()
   else:loss=model_masked_loss(m,inputs,targets,mask)
  scaler.scale(loss).backward();return loss
 def update():
  if not graph:opt.zero_grad(set_to_none=True);loss=forward_backward()
  else:
   for p in m.parameters():p.grad.zero_()
   g.replay();loss=static_loss
  norm,skipped=optimizer_step(m,opt,sched,scaler,1.)
  return float(loss.detach()),norm,skipped
 row={'backend':backend,'fused':fused,'graph':graph}
 try:
  if graph:
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):
     opt.zero_grad(set_to_none=True);forward_backward()
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   opt.zero_grad(set_to_none=True);g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):static_loss=forward_backward()
  for _ in range(3):update()
  times=[];updates=[];torch.cuda.reset_peak_memory_stats()
  for _ in range(5):
   torch.cuda.synchronize();t=time.monotonic();updates.append(update());torch.cuda.synchronize();times.append(time.monotonic()-t)
  row.update(seconds=times,updates=updates,peak_allocated=torch.cuda.max_memory_allocated(),status='PASS')
 except Exception as e:row.update(status='FAIL',error=str(e))
 result['variants'].append(row);(r/'graph-screen.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
 del m,opt,sched,scaler,targets,inputs,mask;gc.collect();torch.cuda.empty_cache()
 if graph and 'g' in globals():del g
