import torch,sys,inspect,json,time,gc
from pathlib import Path
sys.path.insert(0,'/content/d1-opt/v2/src')
from scripts import d1_model as dm
from scripts.train_diffusion import TrainConfig,ShardLoader,block_diffusion_mask,build_optimizer,optimizer_step,model_masked_loss
r=Path('/content/d1-opt/v2');torch.set_num_threads(1)
original=dm.gated_delta_chunked
source=inspect.getsource(original)
source=source.replace('        outputs = []','        states, residuals = [], []')
source=source.replace('            output = mm(queries, state) + mm(scores, updates)\n            outputs.append(output)', '            states.append(state)\n            residuals.append(updates)')
source=source.replace('        output = torch.cat(outputs, dim=2)[:, :, :length].transpose(1, 2).contiguous()', '        output = mm(projected_queries, torch.stack(states, dim=2)) + mm(attention, torch.stack(residuals, dim=2))\n        output = output.reshape(batch, heads, -1, value_dim)[:, :, :length].transpose(1, 2).contiguous()')
ns=dict(vars(dm));exec(source,ns);candidate=ns['gated_delta_chunked'];result={'variants':[],'candidate_source':source}
for label,fn in [('control',original),('batched-output',candidate),('control-repeat',original)]:
 dm.gated_delta_chunked=fn;torch.manual_seed(73);torch.cuda.manual_seed_all(73)
 m=dm.D1LM(32000,1024,backend='chunk-tc').cuda();cfg=TrainConfig(bundle=str(r/'src/probe-bundle'),out='unused',device='cuda',amp=True,fused_optimizer=True)
 opt=build_optimizer(m,cfg);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.);scaler=torch.amp.GradScaler('cuda',init_scale=8)
 targets=next(ShardLoader(cfg.bundle,batch_size=4,seed=17,vocab=32000)).cuda();inputs,mask=block_diffusion_mask(targets,32,frontier=31)
 def update():
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):loss=model_masked_loss(m,inputs,targets,mask)
  scaler.scale(loss).backward();norm,skipped=optimizer_step(m,opt,sched,scaler,1.)
  return float(loss.detach()),norm,skipped
 row={'label':label}
 try:
  for _ in range(3):update()
  torch.cuda.reset_peak_memory_stats();times=[];updates=[]
  for _ in range(5):
   torch.cuda.synchronize();t=time.monotonic();updates.append(update());torch.cuda.synchronize();times.append(time.monotonic()-t)
  row.update(status='PASS',seconds=times,updates=updates,peak_allocated=torch.cuda.max_memory_allocated())
 except Exception as e:row.update(status='FAIL',error=str(e))
 result['variants'].append(row);(r/'batch-screen.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
 del m,opt,sched,scaler,targets,inputs,mask;gc.collect();torch.cuda.empty_cache()
