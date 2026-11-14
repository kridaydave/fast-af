#!/usr/bin/env python3
"""Bounded real-T4 correctness and full-model probe jobs; never pretraining."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_model import (D1Config, D1LM, GatedDeltaNet, gated_delta_chunked,
                              gated_delta_reference)
from scripts.train_diffusion import (TrainConfig, ShardLoader, block_diffusion_mask,
    build_optimizer, load_checkpoint, masked_cross_entropy, optimizer_step)


def comparison(expected, actual, relative, absolute, label):
    a, b = expected.detach().float().cpu(), actual.detach().float().cpu()
    if not bool(torch.isfinite(b).all()):
        raise AssertionError(f'{label}: non-finite output')
    error = float(torch.linalg.vector_norm(a - b))
    scale = float(torch.linalg.vector_norm(a))
    result = {'relative_l2': error / max(scale, 1e-12), 'max_abs': float((a - b).abs().max()),
              'reference_l2': scale, 'allowed_relative': relative, 'allowed_absolute_l2': absolute}
    if error > relative * scale + absolute:
        raise AssertionError(f'{label}: {result}')
    return result


def amp_context(enabled):
    return torch.autocast('cuda', dtype=torch.float16) if enabled else nullcontext()


def core_parity():
    results = []
    for length in (1, 31, 32, 33, 65, 129):
        for amp in (False, True):
            torch.manual_seed(100 + length)
            q = F.normalize(torch.randn(1, length, 4, 128), dim=-1)
            k = F.normalize(torch.randn_like(q), dim=-1)
            v = torch.randn(1, length, 4, 256).to(torch.float16 if amp else torch.float32)
            g = -torch.rand(1, length, 4) * (30 if length == 65 else 0.2)
            beta = torch.rand(1, length, 4)
            initial = torch.randn(1, 4, 128, 256)
            cpu = [t.detach().requires_grad_() for t in (q, k, v, g, beta, initial)]
            gpu = [t.detach().cuda().requires_grad_() for t in cpu]
            expected, es = gated_delta_reference(*cpu)
            with amp_context(amp):
                actual, state = gated_delta_chunked(*gpu)
            probes = torch.randn_like(expected), torch.randn_like(es)
            cg = torch.autograd.grad((expected * probes[0]).sum() + (es * probes[1]).sum(), cpu)
            gg = torch.autograd.grad((actual * probes[0].cuda()).sum() + (state * probes[1].cuda()).sum(), gpu)
            result = {'length': length, 'mode': 'amp-fp16-v' if amp else 'fp32',
                      'output': comparison(expected, actual, 5e-4, 2e-5, 'core output'),
                      'state': comparison(es, state, 5e-4, 2e-5, 'core state'),
                      'gradients': {name: comparison(a, b, 3e-3 if amp else 8e-4, 3e-5, name)
                                    for name, a, b in zip(('q','k','v','decay','beta','initial_state'), cg, gg)}}
            results.append(result)
            print(json.dumps({'core_case_passed': {'length': length, 'amp': amp}}), flush=True)
    return results


def layer_parity():
    results = []
    for amp in (False, True):
        torch.manual_seed(19)
        cfg = D1Config()
        cpu = GatedDeltaNet(cfg, 'torch')
        for module in cpu.modules():
            if isinstance(module, torch.nn.Linear): torch.nn.init.normal_(module.weight, std=0.02)
        torch.nn.init.normal_(cpu.o_proj.weight, std=0.02 / math.sqrt(32))
        gpu = GatedDeltaNet(cfg, 'chunk').cuda()
        gpu.load_state_dict(cpu.state_dict())
        prefix = torch.randn(1, 32, 640)
        with torch.no_grad():
            _, cs = cpu(prefix, use_cache=True)
            with amp_context(amp): _, gs = gpu(prefix.cuda(), use_cache=True)
        snapshot = [gs.recurrent.clone(), *(t.clone() for t in gs.conv)]
        x = torch.randn(1, 33, 640, requires_grad=True)
        y = x.detach().cuda().requires_grad_()
        expected, es = cpu(x, cs, use_cache=True)
        with amp_context(amp): actual, state = gpu(y, gs, use_cache=True)
        probe, sp = torch.randn_like(expected), torch.randn_like(es.recurrent)
        cpu_loss = (expected * probe).sum() + 0.1 * (es.recurrent * sp).sum()
        gpu_loss = (actual.float() * probe.cuda()).sum() + 0.1 * (state.recurrent * sp.cuda()).sum()
        cpu_loss.backward()
        optimizer = torch.optim.SGD(gpu.parameters(), lr=0)
        scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=8192)
        scale = scaler.get_scale()
        scaler.scale(gpu_loss).backward()
        scaler.unscale_(optimizer)
        relative = 0.05 if amp else 0.003
        result = {'mode': 'amp-fp16' if amp else 'fp32',
                  'output': comparison(expected, actual, relative, 1e-6, 'layer output'),
                  'state': comparison(es.recurrent, state.recurrent, relative, 1e-6, 'layer state'),
                  'conv': [comparison(a,b,relative,1e-6,'conv') for a,b in zip(es.conv,state.conv)],
                  'input_gradient': comparison(x.grad, y.grad / scale, relative, 1e-6, 'layer input gradient'),
                  'parameter_gradients': {}}
        for (name, a), (_, b) in zip(cpu.named_parameters(), gpu.named_parameters()):
            assert a.grad is not None and b.grad is not None, name
            result['parameter_gradients'][name] = comparison(a.grad,b.grad,relative,1e-6,name)
        assert all(torch.equal(a,b) for a,b in zip(snapshot,(gs.recurrent,*gs.conv)))
        result['prefix_cache_unchanged'] = True
        scaler.step(optimizer)
        scaler.update()
        results.append(result)
        print(json.dumps({'layer_case_passed': {'amp':amp}}), flush=True)
        del cpu, gpu, optimizer, expected, actual, es, state, cs, gs
        gc.collect(); torch.cuda.empty_cache()
    return results


def full_parity():
    torch.manual_seed(23)
    cpu = D1LM(32000, 1024).eval()
    gpu = D1LM(32000, 1024, backend='chunk').cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    tokens = torch.randint(5,32000,(1,96))
    with torch.no_grad(): expected = cpu(tokens)
    results = []
    for amp in (False, True):
        with torch.no_grad(), amp_context(amp):
            actual = gpu(tokens.cuda())
            entry = gpu.prefill(tokens[:, :32].cuda())
            snapshot = [s.recurrent.clone() if hasattr(s,'recurrent') else s.keys.clone() for s in entry.layers]
            gpu.forward_block(torch.zeros(1,32,dtype=torch.long,device='cuda'),entry)
            cached, committed = gpu.forward_block(tokens[:,32:64].cuda(),entry)
            tail, _ = gpu.forward_block(tokens[:,64:].cuda(),committed)
        relative = 0.03 if amp else 0.003
        results.append({'mode': 'amp-fp16' if amp else 'fp32',
                        'cpu_output':comparison(expected,actual,relative,1e-5,'full CPU logits'),
                        'cached':comparison(actual[:,32:64],cached,0.003,1e-5,'cached logits'),
                        'committed':comparison(actual[:,64:],tail,0.003,1e-5,'committed logits')})
        assert all(torch.equal(a, s.recurrent if hasattr(s,'recurrent') else s.keys)
                   for a,s in zip(snapshot,entry.layers))
        print(json.dumps({'full_forward_passed':{'amp':amp}}),flush=True)
    # Full 109M parameter gradients against CPU reference on a masked 32-token block.
    targets = tokens[:,:32]
    noisy = targets.clone(); noisy[:,::2] = 0
    cpu.train(); cpu.zero_grad(set_to_none=True)
    output = cpu(noisy)
    loss = F.cross_entropy(output[:,::2].flatten(0,1), targets[:,::2].flatten())
    loss.backward()
    for amp in (False,True):
        gpu.train(); gpu.zero_grad(set_to_none=True)
        optimizer = torch.optim.SGD(gpu.parameters(),lr=0)
        scaler = torch.amp.GradScaler('cuda',enabled=amp,init_scale=8192)
        with amp_context(amp):
            logits = gpu(noisy.cuda())
            actual_loss = F.cross_entropy(logits[:,::2].flatten(0,1).float(),targets[:,::2].flatten().cuda())
        scaler.scale(actual_loss).backward(); scaler.unscale_(optimizer)
        relative = 0.05 if amp else 0.003
        gradients={}
        for (name,a),(_,b) in zip(cpu.named_parameters(),gpu.named_parameters()):
            assert a.grad is not None and b.grad is not None,name
            gradients[name]=comparison(a.grad,b.grad,relative,1e-7,name)
        results.append({'mode':'amp-fp16' if amp else 'fp32','masked_ce_cpu':float(loss.detach()),
                        'masked_ce_cuda':float(actual_loss.detach()),'parameter_gradients':gradients})
        scaler.step(optimizer);scaler.update()
        print(json.dumps({'full_backward_passed':{'amp':amp}}),flush=True)
    del cpu,gpu,optimizer,output,logits,expected,actual
    gc.collect();torch.cuda.empty_cache()
    return results


def resume_check(root):
    source=Path(__file__).resolve().parent
    bundle=root/'src'/'probe-bundle'
    base=[sys.executable,'-u',str(source/'train_diffusion.py'),'--bundle',str(bundle),
          '--profile','d1','--gdn-backend','chunk','--device','cuda','--amp',
          '--steps','6','--warmup','1','--batch-size','1','--grad-accum','1',
          '--checkpoint-every','3','--eval-every','3','--eval-batches','1','--verify-hashes']
    def run(name,extra=()):
        result=subprocess.run(base+['--out',str(root/name)]+list(extra),capture_output=True,text=True,timeout=300)
        (root/f'{name}.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,(name,result.stderr)
        return result.stdout
    full=run('resume-full')
    log=root/'resume-interrupted.log'
    child=subprocess.Popen(base+['--out',str(root/'resume-interrupted')],stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True)
    lines=[]
    try:
        for line in child.stdout:
            lines.append(line)
            if line.startswith('{') and json.loads(line).get('step')==4:
                child.kill();break
        rest,_=child.communicate(timeout=30)
        lines.append(rest)
    finally:
        if child.poll() is None:child.kill();child.wait()
        log.write_text(''.join(lines))
    assert child.returncode==-9,child.returncode
    checkpoint=root/'resume-interrupted'/'checkpoint-000003.pt'
    resumed=run('resume-restarted',['--resume',str(checkpoint)])
    def records(text):
        return [row for line in text.splitlines() if line.startswith('{')
                and 'masked_ce' in (row:=json.loads(line))]
    a,b=records(full)[3:],records(resumed)
    assert len(a)==len(b)==3
    errors=[abs(x['masked_ce']-y['masked_ce']) for x,y in zip(a,b)]
    assert max(errors)<1e-3,errors
    left=load_checkpoint(root/'resume-full'/'checkpoint-000006.pt')
    right=load_checkpoint(root/'resume-restarted'/'checkpoint-000006.pt')
    assert left['scheduler']==right['scheduler'] and left['loader']==right['loader']
    assert left['scheduler']['last_epoch']>0,'no successful AMP updates'
    for key in ('step','tokens_seen','tokens_processed','tokens_scored','scaler'):
        assert left[key]==right[key],key
    for key in ('torch','cuda'):
        x,y=left['rng'][key],right['rng'][key]
        if isinstance(x,list):assert all(torch.equal(a,b) for a,b in zip(x,y))
        else:assert torch.equal(x,y)
    metrics={name:comparison(x,right['model'][name],1e-3,1e-5,name) for name,x in left['model'].items()}
    bitwise=all(torch.equal(v,right['model'][k]) for k,v in left['model'].items())
    optimizer_max=0.0
    for index,state in left['optimizer']['state'].items():
        for key,value in state.items():
            other=right['optimizer']['state'][index][key]
            metric=comparison(value,other,1e-3,1e-5,f'optimizer {index} {key}')
            optimizer_max=max(optimizer_max,metric['max_abs'])
    return {'params':json.loads(full.splitlines()[-1])['params'],'context':1024,'amp':True,
            'global_batch':1,'step':6,'kill_trigger_step':4,'killed_exit_code':child.returncode,
            'resume_step':3,'tokens_seen':left['tokens_seen'],'tokens_processed':left['tokens_processed'],
            'tokens_scored':left['tokens_scored'],'successful_updates':left['scheduler']['last_epoch'],
            'loss_max_abs_error':max(errors),'weights_bitwise_equal':bitwise,
            'weights_max_abs_error':max(m['max_abs'] for m in metrics.values()),
            'optimizer_max_abs_error':optimizer_max,'state_metrics':metrics,
            'loader_scheduler_scaler_rng_equal':True}


def probe(root):
    # Explicit limits: at most 24 updates and 300 seconds of measured/probe work.
    for phase in ('parity','resume'):
        evidence=json.loads((root/f'{phase}.json').read_text())
        assert evidence['status']=='PASS' and evidence['gpu']=='Tesla T4',phase
        actual=hashlib.sha256(Path(__file__).with_name('d1_model.py').read_bytes()).hexdigest()
        assert evidence['source_sha256']['d1_model.py']==actual,'stale parity evidence'
    started=time.monotonic()
    cfg=TrainConfig(bundle=str(root/'src'/'probe-bundle'),out=str(root/'probe-unused'),
                    profile='d1',mask_mode='blockwise',gdn_backend='chunk',device='cuda',amp=True,
                    batch_size=1,grad_accum=64,steps=100,warmup=2,lr=1e-3,eval_every=0)
    torch.manual_seed(73);torch.cuda.manual_seed_all(73)
    model=D1LM(32000,1024,backend='chunk').cuda().train()
    optimizer=build_optimizer(model,cfg)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0)
    scaler=torch.amp.GradScaler('cuda',init_scale=8192)
    result={'params':sum(p.numel() for p in model.parameters()),'context':1024,'global_batch':64,
            'backend':'chunk','precision':'fp16 autocast; FP32 delta state/solve and optimizer',
            'initial_scale':8192,'max_updates':24,'wall_cap_seconds':300,'screens':[],'windows':[]}
    updates=0
    def update(micro,mode):
        nonlocal updates
        if updates>=24 or time.monotonic()-started>=300:raise TimeoutError('bounded probe cap reached')
        local=replace(cfg,batch_size=micro,grad_accum=64//micro)
        loader=ShardLoader(local.bundle,batch_size=micro,seed=17,vocab=32000)
        # Match trainer: draw all corruption before forwards; weight by scored tokens.
        batches=[];seen=processed=scored=0
        for _ in range(local.grad_accum):
            targets=next(loader).cuda()
            inputs,mask=block_diffusion_mask(targets,32,frontier=31 if mode=='full-context' else None)
            batches.append((inputs,targets[:,:inputs.shape[1]],mask,int(mask.sum())))
            seen+=targets.numel();processed+=inputs.numel();scored+=int(mask.sum())
        optimizer.zero_grad(set_to_none=True)
        loss_sum=0.0
        for inputs,targets,mask,count in batches:
            with torch.autocast('cuda',dtype=torch.float16):
                loss=masked_cross_entropy(model(inputs),targets,mask)
            assert bool(torch.isfinite(loss)),'nonfinite probe loss'
            scaler.scale(loss.float()*count/scored).backward()
            loss_sum+=float(loss.detach())*count
        norm,skipped=optimizer_step(model,optimizer,scheduler,scaler,1.0)
        updates+=1
        return {'consumed':seen,'processed':processed,'scored':scored,'skipped':int(skipped),
                'masked_ce':loss_sum/scored,'grad_norm':norm}
    def timed_update(micro,mode):
        torch.cuda.synchronize();start=time.monotonic()
        r=update(micro,mode)
        torch.cuda.synchronize();r['seconds']=time.monotonic()-start
        return r
    for micro in (1,2,4,8):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            sample=timed_update(micro,'full-context')
            memory=torch.cuda.max_memory_allocated()
            result['screens'].append({'micro_batch':micro,'status':'PASS','peak_allocated_bytes':memory,
                                      'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**sample})
            print(json.dumps({'screen':result['screens'][-1]}),flush=True)
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            result['screens'].append({'micro_batch':micro,'status':'OOM'})
            break
    usable=[r for r in result['screens'] if r['status']=='PASS' and r['peak_allocated_bytes']<0.85*torch.cuda.get_device_properties(0).total_memory]
    assert usable,'no micro-batch fits with safety margin'
    chosen=max(usable,key=lambda r:r['micro_batch'])['micro_batch']
    result['selected_micro_batch']=chosen;result['grad_accum']=64//chosen
    try:
        for mode in ('full-context','uniform-frontier'):
            for _ in range(2):update(chosen,mode)
            for window in range(3):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                rows=[update(chosen,mode) for _ in range(2)]
                torch.cuda.synchronize();seconds=time.monotonic()-start
                totals={key:sum(r[key] for r in rows) for key in ('consumed','processed','scored','skipped')}
                measured={'mode':mode,'window':window,'updates':2,'seconds':seconds,**totals,
                          'consumed_tokens_per_second':totals['consumed']/seconds,
                          'processed_tokens_per_second':totals['processed']/seconds,
                          'scored_tokens_per_second':totals['scored']/seconds,
                          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                assert not totals['skipped'],'AMP skipped probe updates; not a useful-update measurement'
                result['windows'].append(measured)
                print(json.dumps({'window':measured}),flush=True)
    except TimeoutError as error:
        result['limit_reached']=str(error)
    result.update(updates_run=updates,wall_seconds=time.monotonic()-started,
                  validation_overhead_included=False,checkpoint_overhead_included=False,
                  data_scope='single verified real-token shard subset, cached local I/O; no long run')
    assert result['windows'],'no completed timing windows'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('parity','resume','probe'),required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    report={'phase':args.phase,'status':'RUNNING','python':platform.python_version(),
            'torch':torch.__version__,'cuda':torch.version.cuda,'checks':{}}
    try:
        assert torch.cuda.is_available(),'CUDA unavailable'
        report.update(gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),
                      memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        assert report['gpu']=='Tesla T4' and tuple(report['capability'])==(7,5),report
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        report['source_sha256']={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                                 for f in ('d1_model.py','train_diffusion.py','check_gdn_t4.py')}
        if args.phase=='parity':
            report['checks']['core']=core_parity()
            report['checks']['layer']=layer_parity()
            report['checks']['full_model']=full_parity()
        elif args.phase=='resume': report['checks']['resume']=resume_check(args.root)
        else: report['checks']['probe']=probe(args.root)
        report['status']='PASS'
        code=0
    except Exception as error:
        import traceback
        report.update(status='FAIL',error=str(error),traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
        code=1
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'report':str(args.out)}),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
