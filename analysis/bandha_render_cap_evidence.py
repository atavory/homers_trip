"""Generate manuscript tables and figures from checked capped-error results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS={"generator":"Generator metadata", "lexical":"Lexical", "lexical_metadata":"Lexical + metadata",
        "lettuce":"LettuceDetect", "lettuce_mean":"LettuceDetect mean"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--paper",type=Path,required=True);a=ap.parse_args()
    report=json.loads((a.results/"results.json").read_text());names=report["methods"]
    gen=a.paper/"generated/cap_evidence";figdir=a.paper/"figures/cap_evidence"
    gen.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    rows=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Procedure & Word F1 & $\Pr(E>0)$ & Mean $E$ & Regret [bounds]\\",r"\midrule"]
    for n in names:
        r=report["summaries"][n];lo,hi=r["regret_simultaneous_interval"]
        rows.append(f"{LABELS[n]} & {r['word_f1']:.3f} & {r['error_containing_fraction']:.3f} & {r['mean_remaining']:.3f} & {r['regret']:.3f} [{lo:.3f}, {hi:.3f}]"+r"\\")
    rows.extend([r"\bottomrule",r"\end{tabular}"])
    (gen/"readouts.tex").write_text("\n".join(rows)+"\n")
    vals=[r"\newcommand{\CapResponses}{"+str(report['responses'])+"}",r"\newcommand{\CapSources}{"+str(report['sources'])+"}",r"\newcommand{\CapDraws}{"+str(report['draws'])+"}"]
    for n in names:
        r=report['summaries'][n];tag=''.join(x.capitalize() for x in n.split('_'))
        for suffix,value in [('Regret',r['regret']),('RegretLow',r['regret_simultaneous_interval'][0]),('RegretHigh',r['regret_simultaneous_interval'][1]),('Mean',r['mean_remaining']),('ErrorRate',r['error_containing_fraction'])]:
            vals.append(r"\newcommand{\Cap"+tag+suffix+"}{"+f"{value:.3f}"+"}")
    (gen/"values.tex").write_text("\n".join(vals)+"\n")
    z=np.load(a.results/"curves.npz");x=np.arange(1,report['maximum_error_count']+1)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8,3.1))
    targets=[('generator','lexical_metadata'),('lexical','lexical_metadata')] if 'lettuce' not in names else [('generator','lexical_metadata'),('lettuce','lettuce_mean')]
    for ax,(aa,bb) in zip(axes,targets):
        i,j=names.index(aa),names.index(bb);pairs=z['pairs'].tolist()
        if i<j:
            p=pairs.index([i,j]);d=z['differences'][p];lo,hi=z['lower'][p],z['upper'][p]
        else:
            p=pairs.index([j,i]);d=-z['differences'][p];lo,hi=-z['upper'][p],-z['lower'][p]
        ax.fill_between(x,lo,hi,color='#2a7f95',alpha=.18,label='Joint 95% bootstrap bounds')
        ax.plot(x,d,color='#126779',lw=1.5,label='Capped-error difference')
        ax.axhline(0,color='#555555',lw=.7);ax.set_xscale('log',base=2)
        ax.set_xticks([1,4,16,64,256]);ax.set_xticklabels(['1','4','16','64','256'])
        ax.set_xlabel('Error cap $t$');ax.set_title(LABELS[aa]+' minus\n'+LABELS[bb],fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Expected capped-error difference')
    axes[0].legend(frameon=False,fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir/'cap_bands.pdf',bbox_inches='tight');fig.savefig(figdir/'cap_bands.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Rendered',len(names),'procedures')

if __name__=='__main__':main()
