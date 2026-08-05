#!/usr/bin/env python
"""Perplexity-style evaluation of a nanoGPT checkpoint: WikiText and LAMBADA.

Deliberately separate from eval.py. The multiple-choice suite there (ARC,
BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, SocialIQa, WinoGrande) is scored one
way; these two tasks are scored another way and belong in their own script:

  wikitext        output_type: loglikelihood_rolling. A whole Wikipedia article
                  is cut into windows and scored end to end, so it needs
                  loglikelihood_rolling, which eval.py's NanoGPTLM does not
                  implement. Reports word_perplexity / byte_perplexity /
                  bits_per_byte (lower is better).
                  Data: EleutherAI/wikitext_document_level, wikitext-2-raw-v1,
                  test split (62 articles) - the document-level repackaging of
                  the corpus from Merity et al., Pointer Sentinel Mixture Models.

  lambada_openai  output_type: loglikelihood. Reports acc plus a perplexity of
                  the final word (not per token). It would run under eval.py
                  unchanged; it lives here because the number reported beside
                  WikiText is a perplexity.

Only the scoring is new. Loading the checkpoint (GPTConfig compatibility, the
torch.compile prefix, hyper_conn_n inference, tokenizer choice) is inherited
from eval.NanoGPTLM instead of copied, so a change to GPTConfig still only has
to be handled in one place.

usage:
    python eval_ppl.py --ckpt out-finewebedu-large-predyt/ckpt.pt \
        --output_path out-finewebedu-large-predyt/eval_ppl/ppl.json
"""

import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window, make_table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import NanoGPTLM

TASKS_DEFAULT = "wikitext,lambada_openai"

# Reported per task; anything else the harness returns is kept in the JSON but
# not printed. Perplexities are shown as is, accuracies as percentages.
REPORT = ["word_perplexity", "byte_perplexity", "bits_per_byte", "perplexity", "acc"]
AS_PERCENT = {"acc", "acc_norm"}


class NanoGPTPerplexityLM(NanoGPTLM):
    """NanoGPTLM plus the rolling-window scoring the perplexity tasks need."""

    def __init__(self, *args, max_seq_len=None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_seq_len is not None and max_seq_len > self.block_size:
            raise ValueError(f"--max_seq_len {max_seq_len} exceeds the model's "
                             f"block_size {self.block_size}")
        self.max_seq_len = max_seq_len or self.block_size
        print(f"[setup] rolling window = {self.max_seq_len} tokens "
              f"(block_size = {self.block_size})")

    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        """One total logprob per document.

        A document is usually far longer than the context, so it is cut up with
        the harness' own helpers, exactly as HFLM does:

          get_rolling_token_windows(context_len=1) -> windows that each predict
          a fresh block of tokens, carrying a single token of context over the
          boundary. The final, shorter window slides back to a full block so the
          tail of the document is not scored on a near-empty context.
          make_disjoint_window                     -> trims the context so that
          no token is ever scored twice.

        Reusing these helpers is what keeps the result comparable with published
        lm-eval numbers; naive fixed chunking would score every boundary token
        on less context and report a worse perplexity.

        The score depends on the window length: a model evaluated at 1024 tokens
        cannot be compared against one evaluated at 2048.
        """
        out = []
        for (string,) in [r.args for r in requests]:
            ids = self.encode(string)
            if not ids:
                out.append(0.0)
                continue

            total = 0.0
            for ctx_ids, cont_ids in map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=ids,
                    prefix_token=self.eot_id,   # so the first token also gets a prediction
                    max_seq_len=self.max_seq_len,
                    context_len=1,
                ),
            ):
                seq = ctx_ids + cont_ids                    # <= max_seq_len + 1 tokens
                x = torch.tensor(seq[:-1], dtype=torch.long, device=self._device)[None]
                y = torch.tensor(seq[1:],  dtype=torch.long, device=self._device)[None]

                # Passing y as targets forces the full-sequence logits branch;
                # with targets=None the model only returns the last position.
                logits = self._fwd(x, y)
                logp = F.log_softmax(logits.float(), dim=-1)[0]   # (T, V), fp32

                n = len(cont_ids)                           # score only the fresh tokens
                tgt = y[0, -n:]
                total += float(logp[-n:].gather(-1, tgt[:, None]).sum().item())

            out.append(total)
        return out


def report(results):
    """Compact table: perplexities as is, accuracies as percentages."""
    rows = []
    for task, metrics in results.get("results", {}).items():
        if not isinstance(metrics, dict):
            continue
        for name in REPORT:
            val = metrics.get(f"{name},none", metrics.get(name))
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                shown = f"{100 * val:.2f}" if name in AS_PERCENT else f"{val:.4f}"
                rows.append((task, name, shown))
    if not rows:
        return

    print()
    print(f"{'task':18s} {'metric':18s} {'value':>10s}")
    print("-" * 48)
    for task, name, shown in rows:
        print(f"{task:18s} {name:18s} {shown:>10s}")
    print("-" * 48)
    print("perplexities: lower is better; acc is a percentage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=TASKS_DEFAULT,
                    help=f"comma-separated, default {TASKS_DEFAULT}. "
                         "lambada_standard is the original Paperno et al. release, "
                         "lambada_openai the variant usually reported.")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N documents per task (for debugging)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="float32 is strongly preferred here: a rolling score sums "
                         "hundreds of windows and low-precision error accumulates")
    ap.add_argument("--max_seq_len", type=int, default=None,
                    help="rolling-window length, defaults to the model's block_size. "
                         "Only smaller values are accepted; state whichever you use "
                         "when reporting, results are not comparable across lengths.")
    ap.add_argument("--meta_path", default=None,
                    help="explicit path to a char-level meta.pkl")
    ap.add_argument("--hyper_conn_n", type=int, default=None,
                    help="required when deep_conn is hc/mhc/mhclite and the value is not the default 4")
    ap.add_argument("--include_path", default=None,
                    help="directory of task YAMLs that override the built-in ones")
    ap.add_argument("--output_path", default=None,
                    help="also write the full results to this JSON file")
    args = ap.parse_args()

    lm = NanoGPTPerplexityLM(args.ckpt, device=args.device, dtype=args.dtype,
                             meta_path=args.meta_path, hyper_conn_n=args.hyper_conn_n,
                             max_seq_len=args.max_seq_len)

    task_manager = TaskManager(include_path=args.include_path) if args.include_path else None

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=0,          # both tasks are zero-shot by definition
        limit=args.limit,
        task_manager=task_manager,
    )

    print(make_table(results))
    report(results)

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results["results"], f, indent=2, ensure_ascii=False, default=str)
        print(f"results -> {args.output_path}")


if __name__ == "__main__":
    main()
