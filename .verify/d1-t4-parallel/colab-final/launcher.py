from pathlib import Path
import subprocess,sys,json,time
r=Path('/content/d1-opt-v3');phases=[]
for phase,cap in [('parity',90),('resume',300),('probe',180)]:
 if phase=='probe':args=['scripts/benchmark_d1_parallel.py','--bundle',str(r/'src/probe-bundle'),'--out',str(r/'uniform.json'),'--backend','chunk-tc','--fused','--micro','8']
 else:args=['scripts/check_gdn_t4.py','--phase',phase,'--backend','chunk-tc','--fused-optimizer','--root',str(r),'--out',str(r/(phase+'.json'))]
 t=time.monotonic()
 with (r/(phase+'.log')).open('w') as log:
  try:code=subprocess.run([sys.executable,'-u',str(r/'src'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,timeout=cap).returncode
  except subprocess.TimeoutExpired:code=124
 phases.append({'phase':phase,'exit_code':code,'seconds':time.monotonic()-t});(r/'phases.json').write_text(json.dumps(phases,indent=2))
 if code:break
(r/'done.json').write_text(json.dumps({'exit_code':code}))
