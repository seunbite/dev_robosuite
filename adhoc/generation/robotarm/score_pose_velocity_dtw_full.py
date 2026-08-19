#!/usr/bin/env python3
"""Replay five generation methods and compare canonical 3-D landmark trajectories."""
from __future__ import annotations
import csv, itertools, json, os, sys, tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[3]
RA=REPO/'adhoc/generation/robotarm'; sys.path.insert(0,str(RA))
import run_embodiment_baseline_traj as base
from drag_draw import program_to_xyzrpy

OUT=REPO/'data/results/evaluation/motion_similarity_3d'
TRACE_OUT=OUT/'canonical_traces'
ALL_ROBOTS=('humanoid','tiago','stretch','single_arm','roomba')
ROBOTS=tuple(x for x in os.environ.get('ONLY_ROBOTS',','.join(ALL_ROBOTS)).split(',') if x)
IDX_RAW=os.environ.get('ONLY_IDXS','')
IDXS=tuple(
 int(part)
 for token in IDX_RAW.split(',')
 if token.strip()
 for part in (
  range(int(token.split('-',1)[0]),int(token.split('-',1)[1])+1)
  if '-' in token else (int(token),)
 )
)
if not IDXS: IDXS=tuple(range(1,31))
METHODS=('joint','xyzrpy','motion_program','trajectory','gt')
LABEL={'joint':'Joint','xyzrpy':'Cartesian','motion_program':'Motion Program','trajectory':'Trajectory Sketching','gt':'GT'}
ORIENTATION_LANDMARKS={'base','torso','head','wrist','hand','right_hand','left_hand'}
SCORE_PATH=OUT/os.environ.get('SCORE_FILENAME','pairwise_3d_orientation_scores.csv')

def dedupe_jsonl(path):
 d={}
 for line in path.read_text().splitlines():
  if line.strip():
   r=json.loads(line); d[(r['platform'],int(r['idx']))]=r
 return d

def master(path):
 d={}
 for row in json.loads(path.read_text()):
  for robot,cell in row['platforms'].items(): d[(robot,int(row['idx']))]=cell
 return d

def trace_payload(robot,method,payload,cell,tmp):
 p=json.loads(json.dumps(payload))
 p['skip_capture']=True
 gif=tmp/f'{robot}_{method}.gif'
 base.RENDERERS[robot](p, 'xyzrpy' if method in ('xyzrpy','trajectory') else 'joint', gif, p)
 result={'names':p['_landmark_names'],'xyz':p['_landmarks_xyz'],
         'rot':p['_landmarks_rot'],'hz':int(p.get('hz',8))}
 gif.unlink(missing_ok=True)
 gif.with_suffix('.mp4').unlink(missing_ok=True)
 return result

def trace_path(robot,idx,method):
 return TRACE_OUT/f'{robot}_c{idx:02d}_{method}.npz'

def save_trace(path,trace):
 np.savez_compressed(path,names=np.asarray(trace['names']),xyz=np.asarray(trace['xyz'],float),
                     rot=np.asarray(trace['rot'],float),hz=np.asarray([trace['hz']],int))

def load_trace(path):
 data=np.load(path,allow_pickle=False)
 return {'names':data['names'].tolist(),'xyz':data['xyz'].tolist(),
         'rot':data['rot'].tolist(),'hz':int(data['hz'][0])}

def normalized(t,robot):
 x=np.asarray(t['xyz'],float)
 root=x[:,0].copy(); y=x.copy()
 if robot=='roomba':
  y-=root[0,None,:]; scale=.25
 else:
  y[:,1:]-=root[:,None,:]
  y[:,0]=root-root[0]
  scale=max(float(np.median(np.linalg.norm(x[0,1:]-x[0,0],axis=1))),.15)
 return y/scale

def dtw(a,b,cost):
 d=np.full((len(a)+1,len(b)+1),np.inf); d[0,0]=0
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1): d[i,j]=cost(a[i-1],b[j-1])+min(d[i-1,j],d[i,j-1],d[i-1,j-1])
 return float(d[-1,-1]/(len(a)+len(b)))

def pose_cost(a,b): return float(np.linalg.norm(a-b,axis=1).mean())
def vel_cost(a,b):
 mag=np.linalg.norm(a-b,axis=1)
 na=np.linalg.norm(a,axis=1); nb=np.linalg.norm(b,axis=1)
 active=(na>1e-7)|(nb>1e-7); direction=np.zeros(len(a))
 both=(na>1e-7)&(nb>1e-7)
 direction[active]=1.; direction[both]=1-np.sum(a[both]*b[both],axis=1)/(na[both]*nb[both])
 return float(.5*mag.mean()+.5*direction.mean())

def orientation_trace(t):
 """Keep semantic rigid-body orientations, excluding intermediate arm links."""
 names=np.asarray(t['names'])
 keep=np.array([name in ORIENTATION_LANDMARKS for name in names])
 if not keep.any():
  raise ValueError(f'no orientation landmarks in {list(names)}')
 return np.asarray(t['rot'],float)[:,keep]

def rotation_distance(a,b):
 """Mean geodesic angle (radians) across corresponding link frames."""
 relative=np.matmul(np.swapaxes(a,-1,-2),b)
 trace=np.trace(relative,axis1=-2,axis2=-1)
 return float(np.arccos(np.clip((trace-1.)*.5,-1.,1.)).mean())

def angular_velocity(rot,hz):
 """Framewise local angular velocities in axis-angle coordinates."""
 delta=np.matmul(np.swapaxes(rot[:-1],-1,-2),rot[1:])
 return Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(
     delta.shape[:-2]+(3,)
 )*hz

def compare(a,b,robot):
 pa,pb=normalized(a,robot),normalized(b,robot)
 va=np.diff(pa,axis=0)*a['hz']; vb=np.diff(pb,axis=0)*b['hz']
 ra,rb=orientation_trace(a),orientation_trace(b)
 aa,ab=angular_velocity(ra,a['hz']),angular_velocity(rb,b['hz'])
 return (dtw(pa,pb,pose_cost), dtw(ra,rb,rotation_distance),
         dtw(va,vb,vel_cost), dtw(aa,ab,vel_cost))

def matrix(rows,metric,robot=None):
 z=np.zeros((len(METHODS),len(METHODS))); n=np.zeros((len(METHODS),len(METHODS)),int)
 for i,a in enumerate(METHODS):
  n[i,i]=30 if robot else 150
  for j,b in enumerate(METHODS[i+1:],i+1):
   vals=[r[metric] for r in rows if r['a']==a and r['b']==b and (robot is None or r['robot']==robot)]
   z[i,j]=z[j,i]=np.mean(vals) if vals else np.nan; n[i,j]=n[j,i]=len(vals)
 return z,n

def plot(z,n,title,path):
 fig,ax=plt.subplots(figsize=(6.8,5.8)); finite=z[np.isfinite(z)]
 display_z=z[::-1]; display_n=n[::-1]
 im=ax.imshow(display_z,cmap='Blues',vmin=0,vmax=max(np.percentile(finite,95),.01))
 names=[LABEL[x] for x in METHODS]; ax.set_xticks(range(len(METHODS)),names,rotation=25,ha='right'); ax.set_yticks(range(len(METHODS)),names); ax.set_title(title)
 ax.set_yticklabels(names[::-1])
 threshold=np.nanmax(z)*.55
 for i in range(len(METHODS)):
  for j in range(len(METHODS)):
   s='—' if np.isnan(display_z[i,j]) else f'{display_z[i,j]:.3f}\n(n={display_n[i,j]})'
   ax.text(j,i,s,ha='center',va='center',color='white' if display_z[i,j]>threshold else 'black',fontsize=13.5)
 fig.colorbar(im,ax=ax,label='Distance (lower is closer)'); fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 TRACE_OUT.mkdir(parents=True,exist_ok=True)
 pred={'joint':dedupe_jsonl(REPO/'data/results/baseline_traj/ours5/predictions_joint.jsonl'),
       'xyzrpy':dedupe_jsonl(REPO/'data/results/baseline_traj/ours5/predictions_xyzrpy.jsonl')}
 traj={}
 for path in (
  REPO/'data/results/drag_draw/gemini25pro_c1_25/predictions_rendered.json',
  REPO/'data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_rendered.json',
 ):
  for row in json.loads(path.read_text()):
   if row.get('parse_ok'): traj[(row['platform'],int(row['idx']))]=row
 cfg={'gt':master(REPO/'data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json'),
      'motion_program':master(REPO/'data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json')}
 rows=[]; missing=[]
 with tempfile.TemporaryDirectory(prefix='motion3d_') as td:
  tmp=Path(td)
  for robot in ROBOTS:
   for idx in IDXS:
    traces={}
    for method in METHODS:
     try:
      cache=trace_path(robot,idx,method)
      if cache.is_file():
       traces[method]=load_trace(cache)
       continue
      if method in pred:
       rec=pred[method][(robot,idx)]; payload=rec['payload']; cell=None
      elif method=='trajectory':
       payload=program_to_xyzrpy(traj[(robot,idx)]['program']); cell=None
       if robot in ('humanoid','tiago'):
        payload.update({'ik_priority':'hand_elbow','hand_position_weight':10.0,'elbow_position_weight':10.0,'fingertip_direction_weight':0.5,'wrist_continuity_weight':4.0,'skip_mp4':True})
      else:
       cell=cfg[method][(robot,idx)]; payload=base.EXTRACTORS[robot](cell,hz=8)
      traces[method]=trace_payload(robot,method,payload,cell,tmp)
      save_trace(cache,traces[method])
     except Exception as e:
      missing.append({'robot':robot,'idx':idx,'method':method,'error':repr(e)}); print('[missing]',missing[-1],flush=True)
    for a,b in itertools.combinations(METHODS,2):
     if a in traces and b in traces:
      pose,orient,vel,angular=compare(traces[a],traces[b],robot)
      rows.append({'robot':robot,'idx':idx,'a':a,'b':b,'pose_dtw':pose,
                   'orientation_dtw':orient,'velocity_dtw':vel,
                   'angular_velocity_dtw':angular})
    print(f'[{robot}] {idx:02d}/30 methods={len(traces)}',flush=True)
 if SCORE_PATH.is_file():
  old=list(csv.DictReader(SCORE_PATH.open()))
  selected={(robot,idx) for robot in ROBOTS for idx in IDXS}
  rows=[r for r in old if (r['robot'],int(r['idx'])) not in selected]+rows
  for r in rows:
   r['idx']=int(r['idx'])
   for metric in ('pose_dtw','orientation_dtw','velocity_dtw','angular_velocity_dtw'):
    r[metric]=float(r.get(metric,'nan'))
 rows.sort(key=lambda r:(ALL_ROBOTS.index(r['robot']),int(r['idx']),METHODS.index(r['a']),METHODS.index(r['b'])))
 with SCORE_PATH.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 (OUT/'missing.json').write_text(json.dumps(missing,indent=2)+'\n')
 summary={}
 for metric in ('pose_dtw','orientation_dtw','velocity_dtw','angular_velocity_dtw'):
  for robot in (None,*ALL_ROBOTS):
   z,n=matrix(rows,metric,robot); scope='all' if robot is None else robot
   summary.setdefault(scope,{})[metric]={'distance':z.tolist(),'n':n.tolist()}
   plot(z,n,f'{scope.replace("_"," ").title()} · {metric}',OUT/f'tile_{scope}_{metric}.png')
 (OUT/'matrix_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('DONE',len(rows),'pairs',len(missing),'missing',flush=True)
if __name__=='__main__': main()
