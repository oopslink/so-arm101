#!/usr/bin/env python3
"""Offline analysis only. Never opens a motor port. numpy + matplotlib required.
Training NPZ: export original/future4 action,state,timestamp,episode_index sorted by episode/frame.
"""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--logs',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);a.assets.mkdir(parents=True,exist_ok=True)
joints=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']; quant=360/4095
runs=[]; series=[];manifest=[]
for f in sorted(a.logs.glob('robot-*-control.jsonl')):
 rows=[json.loads(l) for l in f.read_text().splitlines()]; cmd=[r for r in rows if r['kind']=='command'];pos=[r for r in rows if r['kind']=='position'];settle=[r for r in rows if r['kind']=='settling_position'];start=next((r for r in rows if r['kind']=='start'),{})
 manifest.append({'file':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 r={'run':f.name.removesuffix('-control.jsonl'),'commands':len(cmd),'positions':len(pos),'errors':[x['message'] for x in rows if x['kind']=='error']}
 if cmd:
  t=np.array([v['monotonic_ns']/1e9 for v in cmd]);q=np.array([[v['target'][k] for k in joints] for v in cmd]);dt=np.diff(t)
  r.update(start_s=float(t[0]),end_s=float(t[-1]),duration_s=float(t[-1]-t[0]),effective_hz=float(1/dt.mean()) if len(dt) else None,max_interval_ms=float(dt.max()*1000) if len(dt) else None,intervals_over_33ms=int((dt>1/30).sum()),target=start.get('target'),max_joint_change_deg=float(np.max(np.ptp(q,axis=0))),command_increment_p99_deg=float(np.quantile(abs(np.diff(q,axis=0)),.99)) if len(q)>1 else 0)
  nonzero=abs(np.diff(q,axis=0));nonzero=nonzero[nonzero>1e-8];r['nonzero_command_increments_below_encoder_tick_pct']=float((nonzero<quant).mean()*100) if len(nonzero) else None
  if pos:
   tp=np.array([v['monotonic_ns']/1e9 for v in pos]);x=np.array([[v['values'][k] for k in joints] for v in pos]);err=np.array([[v['error'][k] for k in joints] for v in pos]);r['max_tracking_error_deg_by_joint']=dict(zip(joints,abs(err).max(axis=0).tolist()));r['rms_tracking_error_deg_by_joint']=dict(zip(joints,np.sqrt((err**2).mean(axis=0)).tolist()))
   series.append((r,tp,x,t,q))
 if settle:r['settle_range_deg_by_joint']={k:max(v['values'][k] for v in settle)-min(v['values'][k] for v in settle) for k in joints};r['final_position']=settle[-1]['values']
 runs.append(r)
z=np.load(a.training);ep=z['original_episode_index'];episodes=np.unique(ep); durations=[];tr=[]
for e in episodes:
 m=ep==e;t=z['original_timestamp'][m];x=z['original_state'][m,:5];tr.append((int(e),t,x));durations.append(float(t[-1]-t[0]))
# Compare observed states at a common 25 Hz grid, with 200 ms finite-difference windows.
# No cross-episode, cross-segment differences; gripper excluded (different unit).
def smooth_metrics(seqs):
 speeds=[];accels=[];stopped=0;total=0
 for t,x in seqs:
  t=t-t[0];grid=np.arange(0,t[-1]+1e-8,.04)
  if len(grid)<12:continue
  y=np.column_stack([np.interp(grid,t,x[:,j]) for j in range(5)])
  v=(y[5:]-y[:-5])/.2;ac=(v[5:]-v[:-5])/.2
  speeds.append(abs(v));accels.append(abs(ac));total+=len(v);stopped+=int((abs(v).max(axis=1)<1).sum())
 v=np.concatenate(speeds);ac=np.concatenate(accels)
 return {'samples_25hz':total,'all_five_speed_below_1deg_s_pct':100*stopped/total,'speed_abs_p50_by_joint':dict(zip(joints,np.quantile(v,.5,axis=0).tolist())),'speed_abs_p95_by_joint':dict(zip(joints,np.quantile(v,.95,axis=0).tolist())),'accel_abs_p95_by_joint':dict(zip(joints,np.quantile(ac,.95,axis=0).tolist()))}
training=smooth_metrics([(t,x) for e,t,x in tr]);direct=smooth_metrics([(t,x) for r,t,x,tc,q in series if r['max_joint_change_deg']>.5 and not r['errors']])
future_error=max(float(abs(z['future4_action'][ep==e]-z['original_state'][ep==e][np.minimum(np.arange((ep==e).sum())+4,(ep==e).sum()-1)]).max()) for e in episodes)
valid=[r for r in runs if r.get('commands',0)>1];good=[r for r in valid if not r['errors']];gaps=[y['start_s']-x['end_s'] for x,y in zip(valid,valid[1:])];span=valid[-1]['end_s']-valid[0]['start_s'];active=sum(r['duration_s'] for r in valid)
summary={'method':{'common_hz':25,'difference_window_s':.2,'encoder_tick_deg':quant,'gripper_excluded':True,'training_time_is_nominal_not_bus_wall_time':True,'cross_segment_derivatives_excluded':True},'training':{'episodes':len(episodes),'frames':len(ep),'duration_s_min_median_max':np.quantile(durations,[0,.5,1]).tolist(),'state_unchanged_in_future4':bool(np.array_equal(z['original_state'],z['future4_state'])),'future4_formula_max_abs_error':future_error,'original_action_state_abs_p95_by_joint':dict(zip(joints,np.quantile(abs(z['original_action'][:,:5]-z['original_state'][:,:5]),.95,axis=0).tolist())),**training},'direct':{'log_files':len(runs),'command_runs':len(valid),'successful_command_runs':len(good),'error_runs':[r for r in runs if r['errors']],'recorded_span_s':span,'command_span_sum_s':active,'command_span_duty_pct':100*active/span,'inter_run_gap_s_min_median_p95_max':np.quantile(gaps,[0,.5,.95,1]).tolist(),'good_run_hz_min_median_max':np.quantile([r['effective_hz'] for r in good],[0,.5,1]).tolist(),'good_run_max_interval_ms':max(r['max_interval_ms'] for r in good),'good_run_intervals_over_33ms':sum(r['intervals_over_33ms'] for r in good),**direct}}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2));(a.out/'runs.json').write_text(json.dumps(runs,indent=2));(a.out/'log-manifest.json').write_text(json.dumps(manifest,indent=2))
with (a.out/'runs.csv').open('w') as f:
 keys=['run','commands','positions','duration_s','effective_hz','max_interval_ms','max_joint_change_deg','errors'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(runs)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,1,figsize=(12,9),layout='constrained');base=valid[0]['start_s']
for r in valid:axs[0].broken_barh([((r['start_s']-base)/60,r['duration_s']/60)],(0,1),facecolors='#cf583c' if r['errors'] else '#007f86')
axs[0].set(title='Recorded command windows: 50 Hz inside each segment, gaps between segments',xlabel='Minutes since first logged command',yticks=[],ylim=(-.3,1.3))
sel=min(tr,key=lambda s:abs((s[1][-1]-s[1][0])-np.median(durations)));e,t,x=sel
for j in range(5):axs[1].plot(t,x[:,j],label=joints[j])
axs[1].set(title=f'Training follower state: episode {e} (duration nearest median; nominal 30 Hz)',xlabel='Episode time (s)',ylabel='Joint position (deg)');axs[1].legend(ncol=3,fontsize=8)
for r,t,x,tc,q in series:
 if '204418' in r['run']:
  axs[2].plot(tc-tc[0],q[:,1],label='Command, 50 Hz');axs[2].step(t-tc[0],x[:,1],where='post',label='Feedback, about 25 Hz');axs[2].set(title='Large excursion: shoulder 20 to 40 deg (different task/speed from demonstration)',xlabel='Segment time (s)',ylabel='Shoulder lift (deg)');axs[2].legend()
fig.savefig(a.assets/'20260917-continuity-comparison.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');xx=np.arange(5)
for ax,key,title in [(axs[0],'speed_abs_p95_by_joint','95th percentile |velocity| (deg/s)'),(axs[1],'accel_abs_p95_by_joint','95th percentile |acceleration| (deg/s²)')]:
 ax.bar(xx-.18,list(training[key].values()),.36,label='Training follower');ax.bar(xx+.18,list(direct[key].values()),.36,label='Direct control, within segments');ax.set(xticks=xx,xticklabels=['pan','lift','elbow','wrist flex','wrist roll'],title=title);ax.legend(fontsize=8)
fig.suptitle('Common 25 Hz resampling, 200 ms differences; excludes gripper and gaps\nLower magnitude does not establish better perceptual smoothness',fontsize=11)
fig.savefig(a.assets/'20260917-state-dynamics-comparison.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
