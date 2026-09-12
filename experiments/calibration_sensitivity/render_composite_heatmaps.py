from pathlib import Path
import sys,csv,numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
repo=Path('/home/epic-lab/yifu/EPIC-Alignment-calibration-sensitivity')
sys.path.insert(0,str(repo/'experiments/calibration_sensitivity'))
import run_calibration_sensitivity as m

def load(root): return list(csv.DictReader((root/'data'/'heatmap_values.csv').open()))
def matrix(rows,mode,metric):
 out=np.full((len(m.GROUP_ORDER)*len(m.ACCURACIES),len(m.DISPLAY_CONDITIONS)),np.nan)
 gi={g:i for i,g in enumerate(m.GROUP_ORDER)}; ai={a:i for i,a in enumerate(m.ACCURACIES)}; ci={c:i for i,c in enumerate(m.DISPLAY_CONDITIONS)}
 for r in rows:
  if r['alignment_mode']==mode and r['metric']==metric:
   out[gi[r['motion_group']]*len(m.ACCURACIES)+ai[r['accuracy']],ci[r['condition']]]=float(r['mean_relative_change_pct'])
 return out
for dirname in ['calibration_sensitivity_results_pr34_trim_consistent','calibration_sensitivity_results_pr34_no_trim_consistent']:
 root=repo/'experiments/calibration_sensitivity'/dirname; rows=load(root); out=root/'figures'
 mats={(mode,metric):matrix(rows,mode,metric) for mode in m.MODES for metric in m.METRICS}
 extent=max(25.0,max(abs(float(r['mean_relative_change_pct'])) for r in rows)); cmap=LinearSegmentedColormap.from_list('relative_change',('#2B6CB0','#F7F7F7','#C44E3F')); norm=SymLogNorm(linthresh=10,linscale=.9,vmin=-extent,vmax=extent,base=10)
 for metric,metric_name in [('ape','Relative APE change'),('are','Relative ARE change')]:
  fig,axes=plt.subplots(2,2,figsize=(7.16,7.35),squeeze=False)
  fig.subplots_adjust(left=.14,right=.89,bottom=.11,top=.91,wspace=.16,hspace=.30)
  image=None
  short_labels=[f'{g} / {a[0].upper()}' for g in ['Hot3D','EuRoC','KITTI','V2_03','AEA'] for a in m.ACCURACIES]
  for i,(mode,title) in enumerate([
   ('se3_position','SE(3)-original, without calibration'),
   ('se3r_rotation_first','SE3R, without calibration'),
   ('se3_position_calibrated','SE(3)-original, with calibration'),
   ('se3r_rotation_first_calibrated','SE3R, with calibration')]):
   ax=axes.flat[i]; image=m.draw_heatmap(ax,mats[(mode,metric)],metric_name,cmap,norm); ax.set_title(title,loc='left',fontsize=7,fontweight='bold',pad=6)
   if i%2==1: ax.set_yticklabels([])
   else: ax.set_yticklabels(short_labels)
   if i<2: ax.set_xticklabels([])
  cax=fig.add_axes([.91,.20,.018,.60]); m.add_colorbar(fig,cax,image,extent)
  fig.suptitle(f'Calibration sensitivity - {metric_name}',x=.14,y=.965,ha='left',fontsize=9,fontweight='bold')
  stem=f'composite_{metric}'
  fig.savefig(out/f'{stem}.pdf'); fig.savefig(out/f'{stem}.png',dpi=300); fig.savefig(out/f'{stem}.svg'); plt.close(fig)
  print(out/f'{stem}.pdf')
