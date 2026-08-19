#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[3]; ROOT=REPO/'data/results/evaluation/motion_similarity_3d'
ROBOTS=['humanoid','tiago','stretch','single_arm','roomba']; TITLES=['Humanoid','Tiago','Stretch','Single Arm','Roomba']; METHODS=['Joint','Cartesian','Motion Program','Trajectory','GT']
def render(metric,label,stem):
 d=json.loads((ROOT/'matrix_summary.json').read_text()); fig,axes=plt.subplots(1,5,figsize=(16.5,4.3))
 vmax=max(np.nanmax(d[r][metric]['distance']) for r in ROBOTS)
 for col,(robot,title) in enumerate(zip(ROBOTS,TITLES)):
  ax=axes[col];z=np.asarray(d[robot][metric]['distance'],float)[::-1];ax.imshow(z,cmap='Blues',vmin=0,vmax=vmax)
  ax.set_title(title,fontsize=18,fontweight='normal',pad=8);ax.set_xticks(range(5),METHODS,rotation=58,ha='right',fontsize=12)
  ax.set_yticks(range(5),METHODS[::-1] if col==0 else [],fontsize=12);threshold=vmax*.55
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',fontsize=11.25,color='white' if z[i,j]>threshold else '#202020')
  if col==0:ax.set_ylabel(label,fontsize=18)
 fig.tight_layout(pad=.7)
 fig.savefig(ROOT/f'{stem}.png',dpi=220,bbox_inches='tight');fig.savefig(ROOT/f'{stem}.svg',bbox_inches='tight');plt.close(fig)
def main():
 render('pose_dtw','Pose-DTW','paper_pose_dtw_by_embodiment')
 render('velocity_dtw','Velocity-DTW','paper_velocity_dtw_by_embodiment')
if __name__=='__main__':main()
