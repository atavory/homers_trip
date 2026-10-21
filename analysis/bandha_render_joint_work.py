"""Render the unified capped-error/work experiment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bandha_render_cap_evidence import LABELS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--paper',type=Path,required=True);a=ap.parse_args()
    r=json.loads((a.results/'results.json').read_text());names=r['methods']
    gen=a.paper/'generated/joint_work';figdir=a.paper/'figures/joint_work'
    gen.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    rows=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Procedure & Mean $E$ & Opened & $R^E$ & $R^K$ & $R^{\rm joint}$ [bounds]\\',r'\midrule']
    for n in names:
        s=r['summaries'][n];lo,hi=s['joint_bounds']
        rows.append(f"{LABELS[n]} & {s['mean_remaining']:.3f} & {s['expected_opened_responses']:.0f} & {s['error_regret']:.3f} & {s['work_regret']:.3f} & {s['joint_regret']:.3f} [{lo:.3f}, {hi:.3f}]"+r'\\')
    rows.extend([r'\bottomrule',r'\end{tabular}']);(gen/'readouts.tex').write_text('\n'.join(rows)+'\n')
    vals=[]
    for n in names:
        tag=''.join(x.capitalize() for x in n.split('_'));s=r['summaries'][n]
        for suffix,v in [('ErrorRegret',s['error_regret']),('WorkRegret',s['work_regret']),('JointRegret',s['joint_regret']),('JointLow',s['joint_bounds'][0]),('JointHigh',s['joint_bounds'][1]),('Errors',s['mean_remaining']),('Openings',s['expected_opened_responses'])]:
            vals.append(r'\newcommand{\Joint'+tag+suffix+'}{'+f'{v:.3f}'+'}')
    vals.extend([r'\newcommand{\JointResponses}{'+str(r['responses'])+'}',r'\newcommand{\JointSources}{'+str(r['sources'])+'}',r'\newcommand{\JointDraws}{'+str(r['draws'])+'}',r'\newcommand{\JointWorkCaps}{'+str(r['maximum_work_cap'])+'}',r'\newcommand{\JointNoWorkRegret}{'+f"{r['zero_work_sanity_check']['no_work_error_and_joint_regret']:.3f}"+'}',r'\newcommand{\JointWithNoWorkPaidRegret}{'+f"{next(iter(r['zero_work_sanity_check']['positive_work_joint_regrets'].values())):.3f}"+'}'])
    (gen/'values.tex').write_text('\n'.join(vals)+'\n')
    z=np.load(a.results/'curves.npz');i,j=names.index('lettuce'),names.index('lettuce_mean')
    p=z['pairs'].tolist().index([i,j])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for ax,key,title,maximum in zip(axes,['error','work'],['Remaining errors','Checking work'],[r['maximum_error_cap'],r['maximum_work_cap']]):
        x=np.arange(1,maximum+1);d=z[key+'_differences'][p];lo=z[key+'_lower'][p];hi=z[key+'_upper'][p]
        ax.fill_between(x,lo,hi,color='#2a7f95',alpha=.18,label='Joint 95% bootstrap bounds')
        ax.plot(x,d,color='#126779',lw=1.6,label='Full minus mean control')
        ax.axhline(0,color='#555555',lw=.7);ax.set_xscale('log',base=2)
        ax.set_xticks([1,4,16,64,256]);ax.set_xticklabels(['1','4','16','64','256'])
        ax.set_xlabel('Error cap $t$' if key=='error' else 'Work cap $u$')
        ax.set_ylabel('Expected capped-count difference');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=7);fig.tight_layout()
    fig.savefig(figdir/'outcome_bands.pdf',bbox_inches='tight');fig.savefig(figdir/'outcome_bands.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('Rendered error and work outcomes')

if __name__=='__main__':main()
