#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'data/results/evaluation/motion_similarity_3d'
METHODS=['Joint','Cartesian','Motion Program','Trajectory','GT']

def main():
 d=json.loads((ROOT/'matrix_summary.json').read_text())['all']
 fig,axes=plt.subplots(1,2,figsize=(12.8,5.2))
 for ax,(metric,title) in zip(axes,[('pose_dtw','3D Pose-DTW'),('velocity_dtw','3D Velocity-DTW')]):
  z=np.asarray(d[metric]['distance'],float)[::-1]
  im=ax.imshow(z,cmap='Blues',vmin=0,vmax=np.percentile(z,95))
  ax.set_xticks(range(5),METHODS,rotation=24,ha='right'); ax.set_yticks(range(5),METHODS[::-1]); ax.set_title(title,fontweight='normal',fontsize=17,pad=12)
  threshold=np.nanmax(z)*.55
  for i in range(5):
   for j in range(5): ax.text(j,i,f'{z[i,j]:.3f}',ha='center',va='center',fontsize=12,color='white' if z[i,j]>threshold else '#202020')
  cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.set_label('Distance (lower is closer)',fontsize=12)
 fig.tight_layout(pad=.8)
 fig.savefig(ROOT/'paper_pose_velocity_dtw.png',dpi=220,bbox_inches='tight',facecolor='white')
 fig.savefig(ROOT/'paper_pose_velocity_dtw.svg',bbox_inches='tight',facecolor='white')
 plt.close(fig)
if __name__=='__main__': main()
