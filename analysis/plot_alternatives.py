"""Plot the measured word and annotated-span inspection curves."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65))
    budgets = [0.05, 0.1, 0.2]
    policies = [
        ('task_generator', 'conditional', 'Metadata: conditional', '#777777', '-', 'o'),
        ('task_generator', 'global', 'Metadata: global', '#777777', '--', 's'),
        ('lexical', 'conditional', 'Lexical and lexical + metadata: conditional', '#0072B2', '-', 'o'),
        ('lexical', 'global', 'Lexical: global', '#0072B2', '--', 's'),
        ('lexical_metadata', 'global', 'Lexical + metadata: global', '#D55E00', '--', 's'),
    ]
    for ax, outcome, title in zip(axes, ['word_recall', 'complete_span_recall'], ['Word recall', 'Complete annotated-span recall']):
        for model, policy, label, color, style, marker in policies:
            m = result['models'][model]
            keys = [f'{policy}_{outcome}_{b}' for b in budgets]
            point = np.array([m['point'][key] for key in keys])
            lower = np.array([m['confidence95'][key]['low'] for key in keys])
            upper = np.array([m['confidence95'][key]['high'] for key in keys])
            ax.plot(np.array(budgets) * 100, point, color=color, linestyle=style, marker=marker, markersize=4, linewidth=1.3, label=label)
            ax.fill_between(np.array(budgets) * 100, lower, upper, color=color, alpha=0.09)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Nominal inspection budget (%)')
        ax.set_xticks([5, 10, 20])
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8, columnspacing=1.6)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.32, top=0.89, wspace=0.22)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
