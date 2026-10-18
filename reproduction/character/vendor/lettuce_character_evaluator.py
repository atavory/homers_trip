# Extracted verbatim from LettuceDetect; see source_manifest.json and LICENSE_LettuceDetect.
from __future__ import annotations

def evaluate_detector_char_level(
    detector: HallucinationDetector, samples: list[RagTruthSample]
) -> dict[str, float]:
    """
    Evaluate the HallucinationDetector at the character level.

    This function assumes that each sample is a dictionary containing:
      - "prompt": the prompt text.
      - "answer": the answer text.
      - "gold_spans": a list of dictionaries where each dictionary has "start" and "end" keys
                      indicating the character indices of the gold (human-labeled) span.

    It uses the detector (which should have been initialized with the appropriate model)
    to obtain predicted spans, compares those spans with the gold spans, and computes global
    precision, recall, and F1 based on character overlap.

    :param detector: The detector to evaluate.
    :param samples: A list of samples to evaluate.
    :return: A dictionary with global metrics: {"char_precision": ..., "char_recall": ..., "char_f1": ...}
    """
    total_overlap = 0
    total_predicted = 0
    total_gold = 0

    for sample in tqdm(samples, desc="Evaluating", leave=False):
        prompt = sample.prompt
        answer = sample.answer
        gold_spans = sample.labels
        predicted_spans = detector.predict_prompt(prompt, answer, output_format="spans")

        # Compute total predicted span length for this sample.
        sample_predicted_length = sum(
            pred["end"] - pred["start"] for pred in predicted_spans
        )
        total_predicted += sample_predicted_length

        # Compute total gold span length once for this sample.
        sample_gold_length = sum(gold["end"] - gold["start"] for gold in gold_spans)
        total_gold += sample_gold_length

        # Now, compute the overlap between each predicted span and each gold span.
        sample_overlap = 0
        for pred in predicted_spans:
            for gold in gold_spans:
                overlap_start = max(pred["start"], gold["start"])
                overlap_end = min(pred["end"], gold["end"])
                if overlap_end > overlap_start:
                    sample_overlap += overlap_end - overlap_start
        total_overlap += sample_overlap

    precision = total_overlap / total_predicted if total_predicted > 0 else 0
    recall = total_overlap / total_gold if total_gold > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
