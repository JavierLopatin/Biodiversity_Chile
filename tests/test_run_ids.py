"""Distinct settings must produce distinct run ids.

This is the single failure mode that has cost this project the most time, four times over.
`dl_runner.already_done` skips a run whose id already has results, which is what makes the
matrix resumable -- and which means a run id that fails to encode a factor does not error.
It reports `ok` in 0.0 minutes, having computed nothing, and silently returns the *other*
configuration's numbers under this configuration's name.

It happened with `--context`, with `--mixup`, with `--no-augment`, and with `--init-from`,
where three different pretrained checkpoints all mapped to the tag `mae`: two overwrote each
other and the third was skipped.

So: every flag that changes what is fitted gets a case here. Adding a flag to
`scripts/11_run_conv.py` without adding it to `VARIANTS` is the bug.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    """Import `scripts/11_run_conv.py`, whose name is not a valid module identifier."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_conv", ROOT / "scripts" / "11_run_conv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEFAULTS = dict(
    substrate="serpentine", width="B", fusion="late", rotation="trough", normalize="none",
    mixup=False, augment=True, context="clim", arch="sep", row_width=0, aug_slope=0.0,
    aug_prob=1.0, weight_decay=1e-2, p_conv=0.1, p_head=0.3, lr=3e-3, batch_size=64,
    seed_start=0, init_from=None,
)

#: one alternative value per factor. Every one of these must change the run id.
VARIANTS = dict(
    width="C", rotation="none", normalize="zscore", mixup=True, augment=False,
    context="clim+topo+area", arch="res", row_width=4, aug_slope=0.05, aug_prob=0.15,
    weight_decay=0.03, p_conv=0.2, p_head=0.5, lr=1e-3, batch_size=128, seed_start=10,
    init_from="results/mae/mae_serpentine_all_sep_wB_p2_m0.6.pt",
)


def _tags(mod, **over) -> str:
    args = argparse.Namespace(**{**DEFAULTS, **over})
    eff = {k: getattr(args, k) for k in
           ("width", "fusion", "rotation", "normalize", "mixup", "augment")}
    return "_".join(mod._variant_tags(args, eff))


@pytest.fixture(scope="module")
def mod():
    return _load_runner()


@pytest.mark.parametrize("factor", sorted(VARIANTS))
def test_each_factor_changes_the_run_id(mod, factor):
    assert _tags(mod, **{factor: VARIANTS[factor]}) != _tags(mod), \
        f"--{factor.replace('_', '-')} does not appear in the run id; runs will collide " \
        f"and already_done will skip them without computing"


def test_all_single_factor_ids_are_distinct(mod):
    ids = {f: _tags(mod, **{f: v}) for f, v in VARIANTS.items()}
    ids["default"] = _tags(mod)
    dupes = [(a, b) for a, b in itertools.combinations(sorted(ids), 2) if ids[a] == ids[b]]
    assert not dupes, f"factors sharing a run id: {dupes}"


def test_different_checkpoints_get_different_ids(mod):
    """The `mae` case specifically: three checkpoints, three ids."""
    cks = ["results/mae/mae_serpentine_kndvi_sep_wB_p2_m0.6.pt",
           "results/mae/mae_serpentine_all_sep_wB_p2_m0.6.pt",
           "results/mae/mae_serpentine_all_sep_wB_p2_m0.75.pt"]
    ids = [_tags(mod, init_from=c) for c in cks]
    assert len(set(ids)) == 3, f"checkpoints collapsed onto {sorted(set(ids))}"
