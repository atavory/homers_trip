"""Generate the manuscript's conditional editor case and outcome figure."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bandha_editor_frontier import OutcomeModel, upper_curve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--joint-results",type=Path,required=True)
    parser.add_argument("--editor-results",type=Path,required=True)
    parser.add_argument("--operating-results",type=Path,required=True)
    parser.add_argument("--paper",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    joint=json.loads((args.joint_results/"results.json").read_text())
    operating=json.loads((args.operating_results/"results.json").read_text())
    triplets=np.load(args.editor_results/"triplets.npz")
    rows={}
    for name in ["budget_20pct","word_score_0.5"]:
        counts=triplets[name];model=OutcomeModel(counts)
        point=model.evaluate(.9,.02)
        fn,tp,fp=counts.sum(axis=0);flags=int(tp+fp)
        f1=float(2*tp/(fn+tp+flags))
        bound=float((upper_curve(model,.9,.02)-model.reference).max())
        rows[name]={"flags":flags,"word_f1":f1,**point,"upper_B":bound}
    assert rows['budget_20pct']['mean_change']<rows['word_score_0.5']['mean_change']
    assert rows['budget_20pct']['first']>0>rows['word_score_0.5']['first']
    assert rows['word_score_0.5']['upper_B']<0
    assert rows['word_score_0.5']['word_f1']>rows['budget_20pct']['word_f1']
    assert joint['draws']==operating['draws']==5000
    generated=args.paper/'generated/editor_case';figdir=args.paper/'figures/editor_case'
    generated.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Word policy & Flags & F1 & $\Delta\mathbb E[E]$ & $\Delta\Pr(E>0)$ (pp)\\',r'\midrule']
    for name,label in [('budget_20pct',r'Top 20\%'),('word_score_0.5',r'Score $\ge .5$')]:
        row=rows[name]
        table.append(f"{label} & {row['flags']:,} & {row['word_f1']:.3f} & {row['mean_change']:.3f} & {100*row['first']:+.2f}"+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (generated/'readouts.tex').write_text('\n'.join(table)+'\n')
    values=[r'\newcommand{\EditorThresholdUpper}{'+f"{rows['word_score_0.5']['upper_B']:.4f}"+'}']
    for key,tag in [('0.05','Five'),('0.1','Ten'),('0.2','Twenty')]:
        row=operating['budgets'][key]
        for suffix,value in [('Tolerance',row['q_sufficient']),('Low',row['source_interval'][0]),('High',row['source_interval'][1])]:
            values.append(r'\newcommand{\Editor'+tag+suffix+'}{'+f'{value:.4f}'+'}')
    (generated/'values.tex').write_text('\n'.join(values)+'\n')
    curves=np.load(args.joint_results/'curves.npz');names=joint['methods']
    pair=curves['pairs'].tolist().index([names.index('lettuce'),names.index('lettuce_mean')])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    figure,axes=plt.subplots(1,3,figsize=(8,3.1))
    for ax,key,title in zip(axes[:2],['error','work'],['Ideal residual errors','Checking work']):
        x=np.arange(1,curves[key+'_differences'].shape[1]+1)
        ax.fill_between(x,curves[key+'_lower'][pair],curves[key+'_upper'][pair],color='#2a7f95',alpha=.18)
        ax.plot(x,curves[key+'_differences'][pair],color='#126779',lw=1.5)
        ax.axhline(0,color='#555555',lw=.7)
        ax.set_xscale('log',base=2);ax.set_xticks([1,4,16,64,256],['1','4','16','64','256'])
        ax.set_xlabel('Error cap $t$' if key=='error' else 'Work cap $u$')
        ax.set_ylabel('Full minus mean control',fontsize=8);ax.set_title(title,fontsize=9)
    budgets=operating['budgets'];x=100*np.array([float(b) for b in budgets])
    y=100*np.array([v['q_sufficient'] for v in budgets.values()]);bounds=100*np.array([v['source_interval'] for v in budgets.values()])
    axes[2].vlines(x,bounds[:,0],bounds[:,1],color='#126779',lw=1)
    axes[2].plot(x,y,'o-',color='#126779',ms=3,lw=1)
    axes[2].set_xscale('log');axes[2].set_yscale('log');axes[2].set_xticks(x,[str(int(v)) for v in x])
    axes[2].set_xlabel('Word inspection (%)',fontsize=8)
    axes[2].set_ylabel('Sufficient tolerated $q$ (%)',fontsize=8)
    axes[2].set_title(r'Repair probability $\geq .9$',fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(figdir/'outcomes_and_requirements.pdf',bbox_inches='tight')
    figure.savefig(args.out/'outcomes_and_requirements.png',dpi=180,bbox_inches='tight');plt.close(figure)
    result={'model':'Independent item-local editing for the policy table; marginal repair/harm bounds without independence for the sufficient frontier','p':.9,'q':.02,'baseline':'Unedited responses for conditional editing; mean control for ideal score diagnostics','policies':rows,'source_draws':5000,'scope':'Conditional calculations, no executed editor or measured review time. Detection F1 already favors the cutoff policy in this case.','input_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('ideal_results',args.joint_results/'results.json'),('editor_triplets',args.editor_results/'triplets.npz'),('operating_results',args.operating_results/'results.json')]}}
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
