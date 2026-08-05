import os
import sys
import json
import pickle
import argparse
import dataclasses
import tiktoken

import torch
import torch.nn.functional as F

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.gpt_base import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Checkpoint compatibility
#
# ckpt["model_args"] is a snapshot of whatever GPTConfig looked like when the
# run was launched. A checkpoint trained before a field was renamed or removed
# therefore cannot be fed straight into GPTConfig(**model_args). The helpers
# below reconcile the two and print every difference they had to paper over.
# ---------------------------------------------------------------------------

# Old field name -> current field name. Only add a pair here once you have
# confirmed in the git history that the two really denote the same quantity.
# A wrong mapping raises no shape error, it silently changes the forward pass.
CONFIG_RENAMES = {
    # "beta_upper_bound": "angle_upper_bound",
}


def filter_model_args(cfg_cls, margs, verbose=True):
    """Keep only the fields the current cfg_cls declares, applying CONFIG_RENAMES."""
    valid = {f.name for f in dataclasses.fields(cfg_cls)}
    clean, renamed, dropped = {}, [], []

    for k, v in margs.items():
        nk = CONFIG_RENAMES.get(k, k)
        if nk in valid:
            if nk != k:
                renamed.append(f"{k} -> {nk}")
            clean[nk] = v
        else:
            dropped.append(f"{k}={v!r}")

    if verbose:
        if renamed:
            print("[compat] renamed fields: " + ", ".join(renamed))
        if dropped:
            print("[compat] fields present in the checkpoint but not in the current "
                  "GPTConfig, ignored: " + ", ".join(dropped))
            print(f"[compat] that is only safe if they do not affect the forward pass of "
                  f"deep_conn={margs.get('deep_conn')!r}; a strict state_dict load cannot "
                  f"detect this kind of mismatch.")
        defaulted = sorted(valid - set(clean))
        if defaulted:
            print("[compat] fields absent from the checkpoint, current defaults used: "
                  + ", ".join(defaulted))

    return clean


def strip_orig_mod(state_dict):
    """Drop the _orig_mod. prefix left behind by torch.compile (modifies in place)."""
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    return state_dict


def infer_hyper_conn_n(state_dict, verbose=True):
    """Recover hyper_conn_n from the checkpoint's own parameter shapes.

    Nothing is instantiated: the hyper-connection modules already encode the
    stream count in their shapes (static_alpha is (n, n + 1), static_beta is
    (n,)). Building a probe model on the meta device does not work here because
    the hyper_conn initialisation copies out of freshly created tensors, which
    have no storage under a meta context. The strict load_state_dict further
    down is what ultimately confirms the value returned here.
    """
    hc = {k: tuple(v.shape) for k, v in state_dict.items()
          if ".hc_" in k and hasattr(v, "shape")}
    if not hc:
        if verbose:
            print("[compat] no hc_* parameters found in the checkpoint")
        return None

    if verbose:
        seen = set()
        for k, shape in hc.items():
            leaf = k.split(".hc_", 1)[1]
            if leaf not in seen:
                seen.add(leaf)
                print(f"[compat] hc param  hc_{leaf:<32s} {shape}")

    # Strongest signal: the (n, n + 1) mixing matrix, whose shape is unambiguous.
    square = {s[0] for s in hc.values() if len(s) == 2 and s[1] == s[0] + 1}
    if len(square) == 1:
        n = square.pop()
        if verbose:
            print(f"[compat] inferred hyper_conn_n={n} from an (n, n+1) parameter")
        return n

    # Fallback: the shortest 1-D parameter is the per-stream vector of length n.
    # Less certain, so the strict load below matters more in this branch.
    vectors = {s[0] for s in hc.values() if len(s) == 1 and s[0] > 1}
    if vectors:
        n = min(vectors)
        if verbose:
            print(f"[compat] inferred hyper_conn_n={n} from a 1-D parameter "
                  f"(weaker signal, verify against the shapes above)")
        return n

    if verbose:
        print("[compat] hyper_conn_n could not be read off the shapes above")
    return None


def build_codec(ckpt, meta_path_override=None):
    """Pick the tokenizer based on the dataset recorded in the checkpoint.

    data/<dataset>/meta.pkl with stoi/itos -> char-level, otherwise GPT-2 BPE.
    """
    meta_path = meta_path_override
    if meta_path is None:
        dataset = ckpt.get("config", {}).get("dataset")
        if dataset:
            meta_path = os.path.join("data", dataset, "meta.pkl")

    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if "stoi" in meta and "itos" in meta:
            stoi = meta["stoi"]
            print(f"[tokenizer] char-level from {meta_path} (vocab={meta.get('vocab_size')})")
            print("[warn] Standard benchmarks such as HellaSwag/ARC carry little meaning for a "
                  "char-level model; out-of-vocab characters are dropped, so treat the numbers as "
                  "indicative only. For perplexity, refer to the val loss reported by train.py.")

            def encode(s):
                return [stoi[c] for c in s if c in stoi]   # skip chars outside the vocab to avoid KeyError

            return encode, None                            # char-level has no EOT token

    enc = tiktoken.get_encoding("gpt2")
    print("[tokenizer] GPT-2 BPE (tiktoken gpt2)")
    return (lambda s: enc.encode(s, disallowed_special=())), enc.eot_token


@register_model("nanogpt")
class NanoGPTLM(LM):
    """Harness wrapper that only relies on forward(idx, targets) -> (logits, loss),
    so it works for every deep_conn variant in model/gpt_base.py."""

    def __init__(
        self,
        ckpt_path,
        device="cuda",
        dtype="float32",
        meta_path=None,
        hyper_conn_n=None,
        **kwargs
    ):
        super().__init__()

        if "cuda" in device and not torch.cuda.is_available():
            print("[warn] CUDA is unavailable, falling back to CPU")
            device = "cpu"
        # NOTE: some lm_eval versions define LM.device as a read-only property.
        # Store on _device and expose our own property so both old and new
        # versions of the harness work.
        self._device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Strip the _orig_mod. prefix torch.compile leaves behind before anything
        # else, so the hyper_conn_n inference below sees the real parameter names.
        sd = strip_orig_mod(ckpt["model"])

        # Rebuild the config. Fields that have been renamed or removed since this
        # checkpoint was written are filtered out here rather than blowing up in
        # GPTConfig(**margs); see CONFIG_RENAMES at the top of this file.
        margs = filter_model_args(GPTConfig, ckpt["model_args"])
        margs["dropout"] = 0.0                               # no dropout at eval time
        if hyper_conn_n is not None:
            margs["hyper_conn_n"] = hyper_conn_n
        elif margs.get("deep_conn") in ("hc", "mhc", "mhclite") \
                and "hyper_conn_n" not in ckpt["model_args"]:
            # train.py does not persist hyper_conn_n, but the weight shapes pin it down.
            guess = infer_hyper_conn_n(sd)
            if guess is not None:
                margs["hyper_conn_n"] = guess
            else:
                print("[warn] deep_conn is a hyper-connection variant, the checkpoint does not "
                      "store hyper_conn_n and it could not be inferred from the weight shapes, "
                      "so the default of 4 is being used. Pass --hyper_conn_n if training used "
                      "a different value, otherwise the shapes will not match.")

        conf = GPTConfig(**margs)
        model = GPT(conf)

        # Any mismatch that changes a shape still surfaces here.
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] Strict loading failed, retrying with strict=False. This usually means a "
                  f"wrong hyper_conn_n or a difference in the PyTorch flash-attention branch:\n{e}")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print("missing:", missing, "\nunexpected:", unexpected)

        self.model = model.to(self._device).eval()
        self.block_size = conf.block_size
        self._amp_dtype = {"float32": None,
                           "bfloat16": torch.bfloat16,
                           "float16": torch.float16}[dtype]

        self.encode, self.eot_id = build_codec(ckpt, meta_path)
        if self.eot_id is None:
            self.eot_id = 0                                  # dummy prefix for empty contexts

    @property
    def device(self):
        return self._device

    def _fwd(self, x, y):
        if self._amp_dtype is None:
            return self.model(x, y)[0]
        with torch.autocast(device_type=self._device.type, dtype=self._amp_dtype):
            return self.model(x, y)[0]

    @torch.no_grad()
    def loglikelihood(self, requests):
        """Return (sum_logprob, is_greedy) per request; this is all the
        multiple-choice benchmarks need."""
        out = []
        for ctx, cont in [r.args for r in requests]:
            ctx_ids = self.encode(ctx) or [self.eot_id]
            cont_ids = self.encode(cont)
            if len(cont_ids) == 0:
                out.append((0.0, True))
                continue

            ids = (ctx_ids + cont_ids)[-(self.block_size + 1):]
            x = torch.tensor(ids[:-1], dtype=torch.long, device=self._device)[None]
            y = torch.tensor(ids[1:],  dtype=torch.long, device=self._device)[None]

            # Passing y as targets is what forces the full-sequence logits branch;
            # with targets=None the model only returns logits for the last position.
            logits = self._fwd(x, y)
            logp = F.log_softmax(logits.float(), dim=-1)[0]  # (T, V); always run log_softmax in fp32

            n = min(len(cont_ids), logp.size(0))
            tgt = y[0, -n:]
            tok_logp = logp[-n:].gather(-1, tgt[:, None]).squeeze(-1)
            greedy = bool((logp[-n:].argmax(-1) == tgt).all().item())
            out.append((float(tok_logp.sum().item()), greedy))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "Only needed for perplexity-style tasks such as wikitext; train.py already "
            "reports val perplexity.")

    def generate_until(self, requests):
        raise NotImplementedError(
            "Only needed for generative tasks such as gsm8k; multiple-choice benchmarks "
            "never call this.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True,
                    help="comma-separated, e.g. hellaswag,piqa,arc_easy")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N docs per task (for debugging)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="float32 is recommended for loglikelihood comparisons; "
                         "bfloat16 is faster")
    ap.add_argument("--meta_path", default=None,
                    help="explicit path to a char-level meta.pkl")
    ap.add_argument("--hyper_conn_n", type=int, default=None,
                    help="required when deep_conn is hc/mhc/mhclite and the value is not the default 4")
    ap.add_argument("--include_path", default=None,
                    help="directory of task YAMLs that override the built-in ones, "
                         "e.g. ./lm_eval_tasks for the social_iqa parquet fix")
    ap.add_argument("--output_path", default=None,
                    help="also write the full results to this JSON file")
    args = ap.parse_args()

    lm = NanoGPTLM(args.ckpt, device=args.device, dtype=args.dtype,
                   meta_path=args.meta_path, hyper_conn_n=args.hyper_conn_n)

    task_manager = TaskManager(include_path=args.include_path) if args.include_path else None

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        task_manager=task_manager,
    )

    print(make_table(results))

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results["results"], f, indent=2, ensure_ascii=False, default=str)
        print(f"results -> {args.output_path}")


if __name__ == "__main__":
    main()