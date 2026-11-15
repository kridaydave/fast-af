import sys,time,json,gc
from pathlib import Path
import torch
sys.path.insert(0,'/content/d1-opt/src')
from scripts.d1_model import D1LM
from scripts.train_diffusion import TrainConfig,ShardLoader,block_diffusion_mask,build_optimizer,optimizer_step,model_masked_loss
root=Path('/content/d1-opt');torch.set_num_threads(1)
assert torch.cuda.get_device_capability()==(7,5)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda}
a=torch.randn(4,128,128,device='cuda',dtype=torch.float16,requires_grad=True)
b=torch.randn_like(a,requires_grad=True)
try:
 c=torch.bmm(a,b,out_dtype=torch.float32);c.square().mean().backward()
 result['bmm_fp32_output']={'dtype':str(c.dtype),'grad_dtype':str(a.grad.dtype)}
except Exception as e:result['bmm_fp32_output']={'error':str(e)}
torch.manual_seed(73);torch.cuda.manual_seed_all(73)
model=D1LM(32000,1024,backend='chunk').cuda().train()
cfg=TrainConfig(bundle=str(root/'src/probe-bundle'),out='unused',profile='d1',gdn_backend='chunk',amp=True,device='cuda',amp_init_scale=8)
opt=build_optimizer(model,cfg);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
scaler=torch.amp.GradScaler('cuda',init_scale=8)
targets=next(ShardLoader(cfg.bundle,batch_size=4,seed=17,vocab=32000)).cuda()
inputs,mask=block_diffusion_mask(targets,32,frontier=31)
def update():
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):loss=model_masked_loss(model,inputs,targets,mask)
 scaler.scale(loss).backward();norm,skipped=optimizer_step(model,opt,sched,scaler,1.)
 return float(loss.detach()),norm,skipped
for _ in range(3):update()
times=[];updates=[];torch.cuda.reset_peak_memory_stats()
for _ in range(5):
 torch.cuda.synchronize();t=time.monotonic();updates.append(update());torch.cuda.synchronize();times.append(time.monotonic()-t)
result.update(seconds=times,updates=updates,peak_allocated=torch.cuda.max_memory_allocated())
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:update();torch.cuda.synchronize()
rows=p.key_averages()
result['top_ops']=[{'key':r.key,'cuda_us':r.self_device_time_total,'cpu_us':r.self_cpu_time_total,'calls':r.count} for r in sorted(rows,key=lambda r:r.self_device_time_total,reverse=True)[:40]]
result['kernels']=[{'key':r.key,'cuda_us':r.self_device_time_total,'calls':r.count} for r in rows if 'aten::' not in r.key and r.self_device_time_total>500]
(root/'profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
