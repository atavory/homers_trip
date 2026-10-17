"""Verify the optional six-response nonidentification fixture with exact arithmetic."""

import collections
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED_TRUTH = (1, 0, 1, 0, 1, 0)
DESCENDING_SCORES = tuple(F(j, 10) for j in (6, 5, 4, 3, 2, 1))
RANK_LABELS = {
    'A': ('100011', '100101', '010101', '010110', '001110', '111000'),
    'B': ('000111', '000111', '011100', '101100', '110010', '110001'),
}


def assign_scores(rank_labels):
    """Realize labels in score order on the same original fixed truth positions."""
    remaining_positions = {
        y: iter([i for i, truth in enumerate(FIXED_TRUTH) if truth == y])
        for y in (0, 1)
    }
    scores = [None] * len(FIXED_TRUTH)
    for label, score in zip(rank_labels, DESCENDING_SCORES):
        scores[next(remaining_positions[label])] = score
    assert all(score is not None for score in scores)
    observed = tuple(FIXED_TRUTH[i] for i in sorted(range(6), key=lambda i: scores[i], reverse=True))
    assert observed == rank_labels
    return tuple(scores)


def row_readout(scores):
    sorted_indices = sorted(range(6), key=lambda i: scores[i], reverse=True)
    labels = tuple(FIXED_TRUTH[i] for i in sorted_indices)
    positive_scores = [score for score, label in zip(scores, FIXED_TRUTH) if label]
    negative_scores = [score for score, label in zip(scores, FIXED_TRUTH) if not label]
    auc = F(sum(p > n for p in positive_scores for n in negative_scores), 9)
    average_precision = sum((F(sum(labels[:i + 1]), i + 1) for i, y in enumerate(labels) if y), F()) / 3
    recovery = [sum(labels[:k]) for k in range(7)]
    selected = set(sorted_indices[:3])
    tp = sum(FIXED_TRUTH[i] for i in selected)
    fp = 3 - tp
    fn = 3 - tp
    iou = F(tp, tp + fp + fn)
    f1 = F(2 * tp, 2 * tp + fp + fn)

    # Compute ideal Spearman directly on original positions, including tied binary-label ranks.
    score_rank = {score: rank for rank, score in enumerate(sorted(scores), start=1)}
    x = [F(score_rank[score]) - F(7, 2) for score in scores]
    y = [F(5 if truth else 2) - F(7, 2) for truth in FIXED_TRUTH]
    numerator = sum((xi * yi for xi, yi in zip(x, y)), F())
    denominator_squared = sum((xi * xi for xi in x), F()) * sum((yi * yi for yi in y), F())
    assert denominator_squared == F(945, 4)
    return {
        'original_position_scores': list(map(str, scores)),
        'labels_in_descending_score_order': ''.join(map(str, labels)),
        'score_mean': sum(scores) / 6,
        'prevalence': F(sum(FIXED_TRUTH), 6),
        'recovery_counts_by_k': recovery,
        'selected_original_positions_k3': sorted(selected),
        'tp': tp, 'fp': fp, 'fn': fn,
        'iou': iou, 'f1': f1,
        'auc': auc, 'average_precision': average_precision,
        'spearman_numerator': numerator,
        'spearman_common_denominator_squared': denominator_squared,
        'remaining_unsupported_units': fn,
        'complete_support_after_perfect_review': fn == 0,
    }


def mean(values):
    return sum(values, F()) / len(values)


def stringify(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: stringify(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [stringify(v) for v in value]
    return value


def main():
    records = {}
    for name, rows in RANK_LABELS.items():
        label_rows = [tuple(map(int, row)) for row in rows]
        assert all(len(row) == 6 and sum(row) == 3 for row in label_rows)
        scores = [assign_scores(row) for row in label_rows]
        pooled = collections.Counter((score, truth) for row in scores for score, truth in zip(row, FIXED_TRUTH))
        readouts = [row_readout(row) for row in scores]
        rank_positive_counts = [sum(row[k] for row in label_rows) for k in range(6)]
        records[name] = {
            'rows': readouts,
            'pooled_law': sorted((score, truth, count) for (score, truth), count in pooled.items()),
            'positive_counts_at_each_rank': rank_positive_counts,
            'mean_recovery_count_by_k': [mean([r['recovery_counts_by_k'][k] for r in readouts]) for k in range(7)],
            'macro_iou_k3': mean([r['iou'] for r in readouts]),
            'macro_f1_k3': mean([r['f1'] for r in readouts]),
            'macro_auc': mean([r['auc'] for r in readouts]),
            'macro_average_precision': mean([r['average_precision'] for r in readouts]),
            'mean_spearman_numerator': mean([r['spearman_numerator'] for r in readouts]),
            'spearman_common_denominator_squared': F(945, 4),
            'complete_support_rate': F(sum(r['complete_support_after_perfect_review'] for r in readouts), 6),
            'pooled_confusion_k3': {k: sum(r[k] for r in readouts) for k in ('tp', 'fp', 'fn')},
            'inspection_work': 18,
            'successful_corrections': sum(r['tp'] for r in readouts),
        }
    a, b = records['A'], records['B']
    retained_equal = (
        'pooled_law', 'positive_counts_at_each_rank', 'mean_recovery_count_by_k',
        'macro_iou_k3', 'macro_f1_k3', 'macro_auc', 'mean_spearman_numerator',
        'spearman_common_denominator_squared', 'pooled_confusion_k3',
        'inspection_work', 'successful_corrections',
    )
    assert all(a[k] == b[k] for k in retained_equal)
    assert a['positive_counts_at_each_rank'] == [3, 3, 2, 4, 3, 3]
    assert all(row['score_mean'] == F(7, 20) and row['prevalence'] == F(1, 2)
               for record in records.values() for row in record['rows'])
    assert [r['recovery_counts_by_k'][3] for r in a['rows']] == [1, 1, 1, 1, 1, 3]
    assert [r['recovery_counts_by_k'][3] for r in b['rows']] == [0, 0, 2, 2, 2, 2]
    assert a['macro_iou_k3'] == F(1, 3)
    assert a['macro_f1_k3'] == F(4, 9)
    assert a['macro_auc'] == F(13, 27)
    assert a['mean_spearman_numerator'] == F(-1, 2)
    assert a['pooled_confusion_k3'] == {'tp': 8, 'fp': 10, 'fn': 10}
    assert (a['complete_support_rate'], b['complete_support_rate']) == (F(1, 6), F(0))
    # Average precision is optional, not included among the equal summaries.
    assert a['macro_average_precision'] != b['macro_average_precision']
    output = {
        'status': 'PASS',
        'scope': 'Optional fixed-budget constructed example; no named detector or real editor rerun.',
        'fixed_original_truth_every_response': FIXED_TRUTH,
        'same_strictly_ordered_score_multiset_every_response': DESCENDING_SCORES,
        'equal_retained_summary_keys': retained_equal,
        'ideal_mean_spearman': {'exact': '-1/(3*sqrt(105))', 'decimal': -1 / (3 * math.sqrt(105))},
        'records': records,
        'source_assumptions': {
            'support_requirement': 'AIS Section 4.1.2, arXiv2112.12870v2 PDF page12: all information fully supported by source.',
            'url': 'https://arxiv.org/pdf/2112.12870v2#page=12',
            'assumptions': [
                'Six interpretable responses with the same fixed unit truth vector + - + - + -.',
                'Three unsupported factual units per response; support is conjunctive over those units.',
                'Three units inspected per response, selected by the same score-only descending rank policy.',
                'Inspection finds and perfectly corrects unsupported selected units; supported units and all unselected content are unchanged.',
                'Corrections preserve the intended answer and introduce no new errors.',
                'Unit work is homogeneous: eighteen inspections and eight corrections for each detector.',
                'Ideal hard-label IoU and binary-label Spearman are averaged uniformly across six responses.',
            ],
            'limits': [
                'This is not equality of every conditional metric: macro AP differs.',
                'The full per-response ranked-label family is different.',
                'Ideal Spearman equality is exact; numerical library accumulation may differ at roundoff level.',
                'Neither reference nor predictions is constant; rounded-constant branches are not activated by these score values.',
                'No claim that every downstream purpose requires complete support.',
                'No causal or architectural conclusion about a trained detector.',
            ],
        },
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = ROOT / 'optional_fixed_budget_results.json'
    path.write_text(json.dumps(stringify(output), indent=2) + '\n')
    print(json.dumps(stringify({
        'status': output['status'], 'results': str(path),
        'mean_recovery_by_k': a['mean_recovery_count_by_k'],
        'macro_iou': a['macro_iou_k3'], 'macro_f1': a['macro_f1_k3'],
        'macro_auc': a['macro_auc'], 'mean_spearman': output['ideal_mean_spearman'],
        'macro_AP_A_B': [a['macro_average_precision'], b['macro_average_precision']],
        'complete_support_A_B': [a['complete_support_rate'], b['complete_support_rate']],
    })))


if __name__ == '__main__':
    main()
