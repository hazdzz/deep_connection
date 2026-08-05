from __future__ import annotations
import torch
import dataclasses
from typing import Any, Dict, Iterable, Mapping, Optional


RENAMES: Dict[str, str] = {
    # "beta_upper_bound": "angle_upper_bound",
}


def filter_model_args(
    cfg_cls,
    margs: Mapping[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    valid = {f.name for f in dataclasses.fields(cfg_cls)}
    clean: Dict[str, Any] = {}
    renamed, dropped = [], []

    for k, v in margs.items():
        nk = RENAMES.get(k, k)
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
            print("[compat] fields present in the checkpoint but missing from the "
                  "current GPTConfig, ignored: " + ", ".join(dropped))
            print("[compat] this is only safe if those fields have no effect on the "
                  "forward pass for deep_conn=%r; strict loading cannot detect this "
                  "kind of mismatch." % margs.get("deep_conn"))
        defaulted = sorted(valid - set(clean))
        if defaulted:
            print("[compat] fields missing from the checkpoint, falling back to the "
                  "current defaults: " + ", ".join(defaulted))

    return clean


def infer_hyper_conn_n(
    gpt_cls,
    cfg_cls,
    clean_margs: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    candidates: Iterable[int] = range(2, 17),
    verbose: bool = True,
) -> Optional[int]:
    want = {k: tuple(v.shape) for k, v in state_dict.items()
            if hasattr(v, "shape")}

    for n in candidates:
        margs = dict(clean_margs)
        margs["hyper_conn_n"] = n
        try:
            with torch.device("meta"):
                model = gpt_cls(cfg_cls(**margs))
            have = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        except Exception as e:
            if verbose:
                print(f"[compat] hyper_conn_n={n} failed to build: "
                      f"{type(e).__name__}: {e}")
            continue
        if all(have.get(k) == s for k, s in want.items()):
            if verbose:
                print(f"[compat] inferred hyper_conn_n={n} "
                      f"(all weight shapes match)")
            return n

    if verbose:
        print("[compat] could not infer hyper_conn_n from the shapes; "
              "please specify it with --hyper_conn_n")
    return None


def strip_orig_mod(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    return state_dict
