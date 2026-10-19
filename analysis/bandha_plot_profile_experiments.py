"""Vector figures from verified profile experiment outputs."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.results / 'results.json').read_text())
    pair = report['comparisons']['lexical_metadata__minus__lexical']
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'dejavuserif',
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))
    for ax, label, color, title, xlabel in zip(axes, ['errors', 'work'], ['#235e87', '#ad5437'],
                                               ['Residual-error profile', 'Work profile'], ['$a$', '$b$']):
        curve = pair['generating_function_contrasts'][label]
        x = np.asarray(curve['parameter']); point = np.asarray(curve['delta'])
        lo = np.asarray(curve['simultaneous_low']); hi = np.asarray(curve['simultaneous_high'])
        factor = 1000 if label == 'errors' else 1
        ax.fill_between(x, factor * lo, factor * hi, color=color, alpha=.18, linewidth=0)
        ax.plot(x, factor * point, color=color, linewidth=1.8)
        ax.axhline(0, color='#444444', linewidth=.8, linestyle='--')
        ax.set_xlabel(xlabel); ax.set_title(title, fontsize=11); ax.set_xlim(0, 1)
        ax.set_ylabel('$G_E^{combined}-G_E^{lexical}$ ($10^{-3}$)' if label == 'errors'
                      else '$G_K^{combined}-G_K^{lexical}$')
        ax.tick_params(labelsize=9)
    fig.tight_layout(); fig.savefig(args.out / 'profile_bands.pdf'); plt.close(fig)
    spec = importlib.util.spec_from_file_location('profile_values', Path(__file__).with_name('bandha_profile_experiments.py'))
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    prices = np.linspace(0, .1, 101); rates = np.linspace(0, .05, 101)
    region = np.zeros((len(rates), len(prices))); point = np.zeros_like(region)
    critical = report['bootstrap']['critical_max_t']
    for i, v in enumerate(rates):
        for j, shared in enumerate(prices):
            value, low, high = api.value_band(pair['errors'], pair['work'], .5, .5, shared, v, critical)
            point[i, j] = value; region[i, j] = 1 if low > 0 else -1 if high < 0 else 0
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.pcolormesh(prices, rates, region, shading='nearest', cmap=ListedColormap(['#f0cfbf', '#ffffff', '#c4dced']), vmin=-1, vmax=1)
    ax.contour(prices, rates, point, levels=[0], colors=['#333333'], linewidths=[1.0])
    ax.set_xlabel('Shared-work price $F$'); ax.set_ylabel('Per-check price $v$')
    ax.set_title('Value comparison, $a=b=1/2$', fontsize=11)
    ax.legend(handles=[Patch(facecolor='#c4dced', label='Combined higher'),
                       Patch(facecolor='#f0cfbf', label='Lexical higher'),
                       Patch(facecolor='white', edgecolor='#cccccc', label='Unresolved')],
              loc='upper right', fontsize=8, framealpha=.95)
    fig.tight_layout(); fig.savefig(args.out / 'value_regions.pdf'); plt.close(fig)
    print(json.dumps({'figures': ['profile_bands.pdf', 'value_regions.pdf']}))


if __name__ == '__main__':
    main()
