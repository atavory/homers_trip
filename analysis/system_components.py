"""Generate centered score components on the fixed seeded response columns."""

import argparse
import csv
import hashlib
import json
import random
from fractions import Fraction as F
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate(value):
    return format(float(value), '.5f').rstrip('0').rstrip('.') or '0'


def number(value):
    text = format(float(value), '.2f')
    return text[1:] if text.startswith('0.') else text


def generate(inputs):
    rng = random.Random(inputs['score_seed'])
    rows, groups = [], []
    for response in inputs['responses']:
        labels = response['labels']
        count = response['token_count']
        assert count == len(labels) and count > 0 and set(labels) <= {0, 1}
        latent = F(response['latent_prevalence'])
        assert 0 <= latent <= 1
        scores = []
        for label in labels:
            noise = F(rng.randrange(-1000, 1001), 1000)
            raw = F(3, 20) + F(9, 20) * latent + F(1, 4) * label + F(1, 5) * noise
            raw = min(F(1), max(F(0), raw))
            scores.append(F(round(1000 * raw), 1000))
        mean = sum(scores) / count
        deviations = [score - mean for score in scores]
        assert sum(deviations) == 0
        prevalence = F(sum(labels), count)
        group = {**response, 'scores': scores, 'mean': mean,
                 'deviations': deviations, 'prevalence': prevalence}
        groups.append(group)
        offset = F(response['response_id'] - 7, 50)
        for token, (label, score, deviation) in enumerate(zip(labels, scores, deviations), 1):
            for variant, result_mean, result_deviation in [
                ('original', mean, deviation), ('mean', mean, F(0)),
                ('offset', mean + offset, deviation), ('offset_mean', mean + offset, F(0))
            ]:
                result_score = result_mean + result_deviation
                rows.append([variant, response['response_id'], token, label,
                             prevalence, result_score, result_mean, result_deviation])
            assert (score + offset) - (mean + offset) == deviation
        assert sum(score + offset for score in scores) / count == mean + offset
    return groups, rows


def render(groups):
    pitch = F(19, 5)
    largest = max(len(group['labels']) for group in groups)
    end_x = pitch * (len(groups) - 1) + F(17, 5)
    center = end_x / 2
    lines = [
        r'\begingroup',
        r'\definecolor{ResponseScore}{RGB}{32,100,170}',
        r'\definecolor{TokenDeviation}{RGB}{180,92,15}',
        r'\begin{tikzpicture}[x=3.5mm,y=3.5mm,line width=.11mm]',
        r'\node[font=\small] at (' + coordinate(center) + ',' + str(largest + 4) +
        r') {$\hat y_{r,t}=\textcolor{ResponseScore}{m_r}+\textcolor{TokenDeviation}{d_{r,t}}$};',
    ]
    legend_y = largest + 2
    legend_x = center - 17
    for offset, fill, text in [
        (0, 'black', r'Hallucinated'),
        (13, 'white', r'Correct'),
    ]:
        x = legend_x + offset
        lines.append(r'\draw[draw=black!60,fill=' + fill + '] (' + coordinate(x) +
                     ',' + str(legend_y) + ') rectangle ++(1,1);')
        lines.append(r'\node[anchor=west,font=\scriptsize] at (' + coordinate(x + F(3, 2)) +
                     ',' + coordinate(F(2 * legend_y + 1, 2)) + ') {' + text + '};')
    x = legend_x + 23
    lines.append(r'\draw[draw=black!60,fill=black!35] (' + coordinate(x) + ',' +
                 str(legend_y) + ') rectangle ++(.4,1);')
    lines.append(r'\node[anchor=west,font=\scriptsize] at (' + coordinate(x + 1) +
                 ',' + coordinate(F(2 * legend_y + 1, 2)) + r') {Observed prevalence $\pi_r$};')

    for index, group in enumerate(groups):
        x = pitch * index
        count = group['token_count']
        zero = x + F(5, 2)
        lines.append(r'\node[font=\scriptsize] at (' + coordinate(x + F(1, 2)) +
                     ',' + coordinate(F(2 * largest + 1, 2)) +
                     r') {$r=' + str(group['response_id']) + '$};')
        lines.append(r'\draw[draw=black!28,densely dotted] (' + coordinate(zero) +
                     ',0) -- (' + coordinate(zero) + ',' + str(count) + ');')
        for token, (label, deviation) in enumerate(zip(group['labels'], group['deviations'])):
            fill = 'black' if label else 'white'
            lines.append(r'\draw[draw=black!55,fill=' + fill + '] (' + coordinate(x) +
                         ',' + str(token) + ') rectangle ++(1,1);')
            endpoint = zero + F(3, 2) * deviation
            if deviation:
                low, high = sorted([zero, endpoint])
                y0 = F(2 * token + 1, 2) - F(9, 50)
                y1 = F(2 * token + 1, 2) + F(9, 50)
                lines.append(r'\fill[TokenDeviation] (' + coordinate(low) + ',' +
                             coordinate(y0) + ') rectangle (' + coordinate(high) +
                             ',' + coordinate(y1) + ');')
        shade = format(float(100 * group['prevalence']), '.3f')
        lines.append(r'\draw[draw=black!55,fill=black!' + shade + '] (' +
                     coordinate(x + F(5, 4)) + ',0) rectangle ++(.35,' + str(count) + ');')
        lines.append(r'\draw[draw=ResponseScore!45,fill=ResponseScore!4] (' +
                     coordinate(x) + ',-5) rectangle ++(1,4);')
        lines.append(r'\fill[ResponseScore] (' + coordinate(x) +
                     ',-5) rectangle ++(1,' + coordinate(4 * group['mean']) + ');')
        lines.append(r'\node[font=\scriptsize,text=ResponseScore] at (' +
                     coordinate(x + F(1, 2)) + r',-5.7) {$' + number(group['mean']) + '$};')
        if index == 0:
            lines.append(r'\node[anchor=east,font=\tiny,text=ResponseScore] at (-.3,-5) {$0$};')
            lines.append(r'\node[anchor=east,font=\tiny,text=ResponseScore] at (-.3,-1) {$1$};')
            lines.append(r'\node[anchor=west,font=\tiny,text=TokenDeviation] at (' +
                         coordinate(zero) + r',-.55) {$d=0$};')

    lines.extend([
        r'\node[font=\scriptsize,text=ResponseScore] at (' + coordinate(center) +
        r',-6.9) {Response means $m_r$ (common scale $0$ to $1$)};',
        r'\node[font=\scriptsize,text=TokenDeviation] at (' + coordinate(center) +
        r',-8.0) {Orange bars: token deviations $d_{r,t}$; left is negative, right is positive};',
        r'\node[font=\small] at (' + coordinate(center) +
        r',-9.5) {$T:\ \textcolor{ResponseScore}{m_r}\mapsto\textcolor{ResponseScore}{m_r},\ '
        r'\textcolor{TokenDeviation}{d_{r,t}}\mapsto0'
        r'\qquad O_c:\ \textcolor{ResponseScore}{m_r}\mapsto\textcolor{ResponseScore}{m_r+c_r},\ '
        r'\textcolor{TokenDeviation}{d_{r,t}}\mapsto\textcolor{TokenDeviation}{d_{r,t}}$};',
        r'\end{tikzpicture}',
        r'\endgroup',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    groups, rows = generate(inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    tex = render(groups)
    (args.out / 'system_diagram.tex').write_text(tex)
    with (args.out / 'components.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['variant', 'response', 'token', 'label', 'prevalence',
                         'score_exact', 'mean_exact', 'deviation_exact'])
        writer.writerows(rows)
    with (args.out / 'response_means.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['response', 'tokens', 'positives', 'prevalence_exact',
                         'mean_exact', 'display_mean'])
        writer.writerows([g['response_id'], g['token_count'], sum(g['labels']),
                          str(g['prevalence']), str(g['mean']), number(g['mean'])]
                         for g in groups)
    verification = {
        'label_seed': inputs['label_seed'], 'score_seed': inputs['score_seed'],
        'input_sha256': digest(args.inputs), 'script_sha256': digest(Path(__file__)),
        'source_diagram_sha256': inputs['source_diagram_sha256'],
        'responses': len(groups), 'tokens': sum(len(g['labels']) for g in groups),
        'positives': sum(sum(g['labels']) for g in groups),
        'checks': {'original_lengths_and_labels_preserved': True,
                   'scores_equal_mean_plus_deviation': True,
                   'deviations_sum_to_zero_exactly': True,
                   'offset_preserves_deviations': True,
                   'projection_and_offset_commute': True},
        'score_construction': 'clip(3/20 + (9/20)*latent_prevalence + label/4 + uniform_noise/5); quantized to 1/1000',
        'scope': 'synthetic score-label illustration, not trained-detector measurements',
        'outputs': {p.name: digest(p) for p in args.out.iterdir() if p.is_file()
                    and p.name != 'verification.json'},
    }
    (args.out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps({key: verification[key] for key in ['responses', 'tokens', 'positives', 'checks']}))


if __name__ == '__main__':
    main()
