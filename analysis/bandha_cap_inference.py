"""Run a pinned LettuceDetect checkpoint offline on the fixed RAGTruth panel."""
import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, local_files_only=True, reference_compile=False,
        attn_implementation="sdpa").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    raw = {r["id"]: r for r in map(json.loads, args.responses.read_text().splitlines())}
    sources = {r["source_id"]: r for r in map(json.loads, args.sources.read_text().splitlines())}
    index = json.loads(args.index.read_text())
    ids = list(range(args.shard, len(index), args.shards))
    if args.limit:
        ids = ids[:args.limit]
    inputs = []
    for i in ids:
        row = raw[index[i]["id"]]
        assert row["quality"] == "good" and row["split"] == "test"
        prompt = sources[row["source_id"]]["prompt"]
        enc = tokenizer(prompt, row["response"], truncation="only_first",
                        max_length=4096, return_offsets_mapping=True)
        positions = [(j, lo, hi) for j, ((lo, hi), side) in
                     enumerate(zip(enc.pop("offset_mapping"), enc.sequence_ids()))
                     if side == 1 and hi > lo]
        matches = list(re.finditer(r"\S+", row["response"]))
        assert len(matches) == index[i]["word_count"]
        inputs.append((i, enc, positions, [(m.start(), m.end()) for m in matches]))
    inputs.sort(key=lambda x: len(x[1]["input_ids"]))
    output = {}
    lengths = []
    for start in range(0, len(inputs), args.batch_size):
        group = inputs[start:start + args.batch_size]
        batch = tokenizer.pad([x[1] for x in group], padding=True, return_tensors="pt")
        with torch.inference_mode():
            probabilities = model(**batch).logits.float().softmax(-1)[..., 1].numpy()
        for bi, (i, enc, positions, spans) in enumerate(group):
            scores = np.zeros(len(spans), dtype=np.float32)
            covered = np.zeros(len(spans), dtype=bool)
            ends = [b for a, b in spans]
            for token, lo, hi in positions:
                j = bisect.bisect_right(ends, lo)
                while j < len(spans) and spans[j][0] < hi:
                    scores[j] = max(scores[j], probabilities[bi, token])
                    covered[j] = True
                    j += 1
            if not covered.all():
                raise ValueError((i, "Unscored answer words", np.flatnonzero(~covered).tolist()))
            output[i] = scores
            lengths.append(len(enc["input_ids"]))
        if start == 0 or (start // args.batch_size) % 10 == 0 or start + args.batch_size >= len(inputs):
            print(json.dumps({"shard": args.shard, "completed": len(output), "total": len(inputs),
                              "elapsed_seconds": round(time.monotonic() - started, 1),
                              "max_batch_tokens": batch["input_ids"].shape[1]}), flush=True)
    ordered = sorted(output)
    offsets = np.r_[0, np.cumsum([len(output[i]) for i in ordered])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, response_indices=ordered, offsets=offsets,
                        scores=np.concatenate([output[i] for i in ordered]))
    report = {"checkpoint": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
              "revision": "bbd77832f52f9bd87546a3924c032467921f5c34",
              "model_sha256": digest(args.model / "model.safetensors"),
              "tokenizer_sha256": digest(args.model / "tokenizer.json"),
              "response_sha256": digest(args.responses), "source_sha256": digest(args.sources),
              "index_sha256": digest(args.index), "scores_sha256": digest(args.out),
              "script_sha256": digest(__file__), "torch": torch.__version__,
              "device": "cpu", "precision": "float32", "max_sequence_length": 4096,
              "truncation": "only_first (prompt)", "word_score": "maximum overlapping answer-subtoken probability",
              "shard": args.shard, "shards": args.shards, "responses": len(output),
              "elapsed_seconds": time.monotonic() - started,
              "maximum_observed_sequence_length": max(lengths)}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
