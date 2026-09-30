"""A plain PyTorch training loop.

`lightning` is not installed in this environment and there is no reason to add it: the loop
below is 80 lines, and being able to read it matters more here than the features it lacks.
`Trait_2DCNN/training/train_ood.py:172-190` already contains an equivalent Lightning-free
path, so this is the same pattern the prior work fell back to.

What the loop guarantees, and why each one matters at n ~ 1,000:

- **Early stopping on a *grouped* inner-validation split**, with best weights restored. A
  random inner split would put same-owner or same-block plots on both sides and choose the
  stopping epoch against a leaky signal, undoing the outer scheme.
- **`drop_last=True`.** BatchNorm on a batch of one raises; with 700 training rows and batch
  64 that happens whenever the remainder is 1.
- **Deterministic seeding** of torch, numpy and the DataLoader workers, so the five-seed
  ensemble measures model variance rather than scheduling noise.
- **Inner-validation residuals are returned**, because Duan's smearing correction needs
  residuals the model did not fit directly (see :mod:`biodiv.targets`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import augment_curves
from .losses import make_loss


@dataclass
class TrainCfg:
    lr: float = 3e-3
    weight_decay: float = 1e-2
    max_epochs: int = 300
    warmup_epochs: int = 10
    batch_size: int = 64
    patience: int = 25
    grad_clip: float = 1.0
    loss: str = "huber"
    huber_delta: float = 1.0
    augment: bool = True
    mixup: bool = False
    mixup_alpha: float = 0.2
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    aug: dict = field(default_factory=lambda: dict(jitter_sd=0.5, amp=0.05,
                                                   baseline=0.02, noise=0.01))
    # Deep CORAL (Sun & Saenko 2016). Off unless ``coral`` is set; then ``train_one_fold``
    # needs ``ds_target``. See ``coral_loss`` for where and how the loss is applied.
    coral: bool = False
    coral_lambda: float = 0.0
    coral_ramp_epochs: int = 10
    coral_bn_update: bool = False


class CurveDataset(Dataset):
    """Holds the source curves and rebuilds the substrate per batch when augmenting.

    ``build`` maps ``(B, C, 52) -> (B, C', H, W)``; for `curve1d` it is the identity. Keeping
    the curve as the stored form is what lets the augmentation be phenological rather than
    pictorial.
    """

    def __init__(self, curves: np.ndarray, images: np.ndarray, ctx: np.ndarray,
                 y: np.ndarray, mask: np.ndarray, patch: np.ndarray | None = None,
                 build=None, augment: bool = False, aug: dict | None = None,
                 seed: int = 0):
        self.curves = curves
        self.images = images
        self.ctx = ctx.astype(np.float32)
        self.y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.patch = patch
        self.build = build
        self.augment = augment and build is not None
        self.aug = aug or {}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.augment:
            c = augment_curves(self.curves[i][None], self.rng, **self.aug)
            img = self.build(c)[0]
        else:
            img = self.images[i]
        out = [torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
               torch.from_numpy(self.ctx[i]),
               torch.from_numpy(self.y[i]),
               torch.from_numpy(self.mask[i])]
        if self.patch is not None:
            out.append(torch.from_numpy(self.patch[i]))
        return tuple(out)


def _lr_lambda(cfg: TrainCfg):
    def f(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        t = (epoch - cfg.warmup_epochs) / max(1, cfg.max_epochs - cfg.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * min(t, 1.0)))
    return f


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def coral_loss(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: squared Frobenius distance between the source and target feature
    covariances, scaled by 1/(4 d^2) as in Sun & Saenko (2016, arXiv:1607.01719)."""
    d = zs.shape[1]
    xm = zs - zs.mean(0, keepdim=True)
    xt = zt - zt.mean(0, keepdim=True)
    return ((xm.T @ xm - xt.T @ xt) ** 2).mean() / (4 * d * d)


class _BNFrozenStats:
    """Context manager: BatchNorm keeps normalising with the batch statistics (train mode)
    but does not update its running statistics. Used on the target-domain pass, so the
    test block does not reach the running stats -- that alone would be AdaBN (Li et al.
    2016), a domain adaptation of its own, and it would happen even at lambda = 0."""

    def __init__(self, model, active: bool):
        self.bns = [m for m in model.modules()
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)] if active else []

    def __enter__(self):
        self.saved = [m.momentum for m in self.bns]
        for m in self.bns:
            m.momentum = 0.0

    def __exit__(self, *exc):
        for m, v in zip(self.bns, self.saved):
            m.momentum = v


def _forward(model, batch, device, use_patch: bool):
    img, ctx, y, m = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
    patch = batch[4].to(device) if use_patch and len(batch) > 4 else None
    pred = model(img, ctx, patch) if use_patch else model(img, ctx)
    return pred, y, m


def train_one_fold(model, ds_fit: CurveDataset, ds_val: CurveDataset, cfg: TrainCfg,
                   seed: int = 0, use_patch: bool = False, verbose: bool = False,
                   ds_target: CurveDataset | None = None):
    """Fit with early stopping. Returns ``(model, history, val_residuals)``.

    With ``cfg.coral``, ``ds_target`` holds the unlabelled predictors of the outer test
    block (transductive). Each source batch is paired with a target batch from a cyclic
    loader, both go through ``model.embed`` (the pooled curve representation, before the
    context is concatenated -- aligning topography and area across blocks would be bias,
    not adaptation), and ``lambda * coral_loss`` is added, with lambda ramped linearly to
    ``cfg.coral_lambda`` over ``cfg.coral_ramp_epochs``. The target pass runs at every
    lambda, including 0, so a lambda = 0 control goes through the same code path. Early
    stopping still uses only the supervised inner-validation loss.

    ``val_residuals`` are (n_val, n_targets) in the fitted (scaled) units, NaN where the
    target was unobserved — the input Duan's smearing needs.
    """
    seed_everything(seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    crit = make_loss(cfg.loss, cfg.huber_delta)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg))

    g = torch.Generator().manual_seed(seed)
    dl_fit = DataLoader(ds_fit, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, generator=g)
    dl_val = DataLoader(ds_val, batch_size=256, shuffle=False,
                        num_workers=cfg.num_workers)
    rng = np.random.default_rng(seed)

    if cfg.coral:
        if ds_target is None or cfg.mixup or use_patch:
            raise ValueError("coral needs ds_target, and is not wired for mixup or patch")
        gt = torch.Generator().manual_seed(seed + 10_000)
        dl_tgt = DataLoader(ds_target, batch_size=cfg.batch_size, shuffle=True,
                            drop_last=True, num_workers=cfg.num_workers, generator=gt)
        it_tgt = iter(dl_tgt)

    best, best_state, bad = np.inf, None, 0
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        tot, ctot, nb = 0.0, 0.0, 0
        lam_c = cfg.coral_lambda * min(1.0, (epoch + 1) / max(1, cfg.coral_ramp_epochs))
        for batch in dl_fit:
            opt.zero_grad(set_to_none=True)
            if cfg.coral:
                try:
                    tb = next(it_tgt)
                except StopIteration:
                    it_tgt = iter(dl_tgt)
                    tb = next(it_tgt)
                img, ctx = batch[0].to(device), batch[1].to(device)
                y, m = batch[2].to(device), batch[3].to(device)
                zs = model.embed(img)
                pred = model.head(torch.cat([zs, ctx], dim=1)) if ctx.shape[1] else model.head(zs)
                with _BNFrozenStats(model, not cfg.coral_bn_update):
                    zt = model.embed(tb[0].to(device))
                lc = coral_loss(zs, zt)
                loss = crit(pred, y, m) + lam_c * lc
                ctot += float(lc)
                nb += 1
            elif cfg.mixup:
                pred, y, m = _forward(model, batch, device, use_patch)
                # Mix the inputs, then mix the two masked losses — never the targets, which
                # carry NaN by design and would poison any convex combination.
                perm = torch.randperm(y.shape[0], device=device)
                lam = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                img = batch[0].to(device)
                mixed = lam * img + (1 - lam) * img[perm]
                ctx = batch[1].to(device)
                mixed_ctx = lam * ctx + (1 - lam) * ctx[perm]
                pred = model(mixed, mixed_ctx)
                loss = lam * crit(pred, y, m) + (1 - lam) * crit(pred, y[perm], m[perm])
            else:
                pred, y, m = _forward(model, batch, device, use_patch)
                loss = crit(pred, y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss) * y.shape[0]
        sched.step()

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for batch in dl_val:
                pred, y, m = _forward(model, batch, device, use_patch)
                vtot += float(crit(pred, y, m)) * y.shape[0]
                vn += y.shape[0]
        vloss = vtot / max(vn, 1)
        history.append(dict(epoch=epoch, train_loss=tot / max(len(ds_fit), 1), val_loss=vloss,
                            lr=opt.param_groups[0]["lr"]))
        if cfg.coral:
            history[-1].update(coral_loss=ctot / max(nb, 1), coral_lambda=lam_c)
        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  train {history[-1]['train_loss']:.4f}  val {vloss:.4f}")

        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    resid = _residuals(model, ds_val, cfg, use_patch)
    return model, history, resid


def train_fixed_epochs(model, ds_fit: CurveDataset, cfg: TrainCfg, seed: int = 0,
                       use_patch: bool = False, verbose: bool = False):
    """Fit for exactly ``cfg.max_epochs`` epochs on ``ds_fit``, no validation split, no
    early stopping. For a deployment model trained on 100% of the data, where there is no
    held-out slice left to early-stop against — the epoch budget must come from outside
    (e.g. the early-stopped epoch count from the CV runs), and ``cfg.max_epochs`` must equal
    that budget so the cosine LR schedule (:func:`_lr_lambda`) actually completes.

    Returns ``(model, history)`` — ``history`` has ``train_loss`` only, no ``val_loss``.
    """
    seed_everything(seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    crit = make_loss(cfg.loss, cfg.huber_delta)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg))

    g = torch.Generator().manual_seed(seed)
    dl_fit = DataLoader(ds_fit, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, generator=g)
    rng = np.random.default_rng(seed)

    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        tot = 0.0
        for batch in dl_fit:
            opt.zero_grad(set_to_none=True)
            pred, y, m = _forward(model, batch, device, use_patch)
            if cfg.mixup:
                perm = torch.randperm(y.shape[0], device=device)
                lam = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                img = batch[0].to(device)
                mixed = lam * img + (1 - lam) * img[perm]
                ctx = batch[1].to(device)
                mixed_ctx = lam * ctx + (1 - lam) * ctx[perm]
                pred = model(mixed, mixed_ctx)
                loss = lam * crit(pred, y, m) + (1 - lam) * crit(pred, y[perm], m[perm])
            else:
                pred, y, m = _forward(model, batch, device, use_patch)
                loss = crit(pred, y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss) * y.shape[0]
        sched.step()
        train_loss = tot / max(len(ds_fit), 1)
        history.append(dict(epoch=epoch, train_loss=train_loss, lr=opt.param_groups[0]["lr"]))
        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  train {train_loss:.4f}")

    return model, history


@torch.no_grad()
def predict(model, ds: CurveDataset, cfg: TrainCfg, use_patch: bool = False) -> np.ndarray:
    model.eval()
    device = torch.device(cfg.device)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=cfg.num_workers)
    out = []
    for batch in dl:
        pred, _, _ = _forward(model, batch, device, use_patch)
        out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def _residuals(model, ds: CurveDataset, cfg: TrainCfg, use_patch: bool) -> np.ndarray:
    pred = predict(model, ds, cfg, use_patch)
    obs = ds.y.copy()
    obs[ds.mask == 0] = np.nan
    return obs - pred
