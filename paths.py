"""
Shared helpers for constructing output / checkpoint paths,
so that train.py and sample.py always agree on the directory layout.
"""


def build_out_dir(out_prefix_dataset, out_prefix_model, out_prefix_method):
    """Construct the checkpoint output directory name."""
    return f"out-{out_prefix_dataset}-{out_prefix_model}-{out_prefix_method}"