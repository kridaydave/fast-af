import sys,time,json,gc,ast,inspect,types,subprocess
from pathlib import Path
import torch
sys.path.insert(0,'/content/d1-opt/src')
from scripts import d1_model as dm
from scripts.train_diffusion import TrainConfig,ShardLoader,block_diffusion_mask,build_optimizer,optimizer_step,model_masked_loss
root=Path('/content/d1-opt');torch.set_num_threads(1)
assert torch.cuda.get_device_capability()==(7,5)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'nvcc':subprocess.run(['which','nvcc'],capture_output=True,text=True).stdout.strip(),'variants':[]}
class HalfMM(torch.autograd.Function):
 @staticmethod
 def forward(ctx,a,b):
  ah,bh=a.half(),b.half();ctx.save_for_backward(ah,bh)
  return torch.bmm(ah,bh,out_dtype=torch.float32)
 @staticmethod
 def backward(ctx,g):
  a,b=ctx.saved_tensors;g=g.half()
  return torch.bmm(g,b.transpose(-1,-2),out_dtype=torch.float32),torch.bmm(a.transpose(-1,-2),g,out_dtype=torch.float32)
def mm(a,b):
 shape=(*a.shape[:-1],b.shape[-1]);return HalfMM.apply(a.reshape(-1,*a.shape[-2:]),b.reshape(-1,*b.shape[-2:])).reshape(shape)
class MMTransform(ast.NodeTransformer):
 def visit_BinOp(self,node):
  self.generic_visit(node)
  if isinstance(node.op,ast.MatMult):return ast.copy_location(ast.Call(func=ast.Name(id='mm',ctx=ast.Load()),args=[node.left,node.right],keywords=[]),node)
  return node
source=ast.fix_missing_locations(MMTransform().visit(ast.parse(inspect.getsource(dm.gated_delta_chunked))))
ns=dict(vars(dm));ns['mm']=mm;exec(compile(source,'<half-mm-experiment>','exec'),ns)
original=dm.gated_delta_chunked;mixed=ns['gated_delta_chunked']
# Independent CPU recurrence, long and strongly decaying sequences, all six grads.
result['core']=[]
for length,strong in ((33,False),(257,False),(1024,False),(65,True)):
 torch.manual_seed(17);q=torch.nn.functional.normalize(torch.randn(1,length,4,128),dim=-1);k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
 args=[q,k,torch.randn(1,length,4,256),-torch.rand(1,length,4)*(30 if strong else .2),torch.rand(1,length,4),torch.randn(1,4,128,256)]
 cpu=[t.requires_grad_() for t in args];gpu=[t.detach().cuda().requires_grad_() for t in args]
 o,s=dm.gated_delta_reference(*cpu);a,b=mixed(*gpu)
 probes=[torch.randn_like(o),torch.randn_like(s)]
 cg=torch.autograd.grad((o*probes[0]).sum()+(s*probes[1]).sum(),cpu)
 gg=torch.autograd.grad((a*probes[0].cuda()).sum()+(b*probes[1].cuda()).sum(),gpu)
 def metric(x,y):
  x,y=x.detach().float().cpu(),y.detach().float().cpu();return {'relative_l2':float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x)),'max_abs':float((x-y).abs().max()),'finite':bool(torch.isfinite(y).all())}
 result['core'].append({'length':length,'strong':strong,'output':metric(o,a),'state':metric(s,b),'grads':[metric(x,y) for x,y in zip(cg,gg)]})
(root/'experiment.json').write_text(json.dumps(result,indent=2));print('core complete',flush=True)
for label,fn,fused in [('baseline',original,False),('half-mm',mixed,False),('half-mm-fused-adam',mixed,True),('baseline-repeat',original,False)]:
 dm.gated_delta_chunked=fn;torch.manual_seed(73);torch.cuda.manual_seed_all(73)
 model=dm.D1LM(32000,1024,backend='chunk').cuda().train()
 cfg=TrainConfig(bundle=str(root/'src/probe-bundle'),out='unused',profile='d1',gdn_backend='chunk',amp=True,device='cuda',amp_init_scale=8)
 opt=build_optimizer(model,cfg)
 if fused:
  opt=torch.optim.AdamW(opt.param_groups,lr=cfg.lr,betas=(cfg.beta1,cfg.beta2),eps=cfg.eps,fused=True)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
 scaler=torch.amp.GradScaler('cuda',init_scale=8)
 targets=next(ShardLoader(cfg.bundle,batch_size=4,seed=17,vocab=32000)).cuda()
 inputs,mask=block_diffusion_mask(targets,32,frontier=31)
 def update():
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):loss=model_masked_loss(model,inputs,targets,mask)
  scaler.scale(loss).backward();norm,skipped=optimizer_step(model,opt,sched,scaler,1.)
  return float(loss.detach()),norm,skipped
 row={'variant':label}
 try:
  for _ in range(3):update()
  times=[];updates=[];torch.cuda.reset_peak_memory_stats()
  for _ in range(5):
   torch.cuda.synchronize();t=time.monotonic();updates.append(update());torch.cuda.synchronize();times.append(time.monotonic()-t)
  row.update(seconds=times,updates=updates,peak_allocated=torch.cuda.max_memory_allocated(),status='PASS')
 except Exception as e:row.update(status='FAIL',error=str(e))
 result['variants'].append(row);(root/'experiment.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
 del model,opt,sched,scaler,targets,inputs,mask;gc.collect();torch.cuda.empty_cache()
