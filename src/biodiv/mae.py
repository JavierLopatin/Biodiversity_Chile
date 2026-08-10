"""Masked autoencoding on the pixel curves, to pretrain the 2-D trunk.

The convolutional family's problem is not architecture, it is 1,082 labelled plots. But the
same cubes hold **135,250 unlabelled pixel curves** -- 1,082 plots x 25 pixels x 5 indices --
from the same sensor, the same window and the same processing. That is 125x more data than
there are labels, and a masked autoencoder is the standard way to spend it.

**Why not the ViT from `Trait_2DCNN/models/mae_2d.py`.** That one is a ViT-Tiny at 224x224
with patch 16, roughly 2.7M parameters. Our images are 8x8. Shrinking the patch to 2 leaves
16 tokens, and a 6-layer transformer over 16 tokens is a lot of machinery for very little
sequence -- and far outside the <60k budget the project already measured as the right regime
(width X, 788k parameters, loses).

So the encoder here **is the existing `PhenoNetS` trunk**. That is the point: fine-tuning
produces exactly the supervised model with different initial weights, so the comparison
isolates pretraining and nothing else. A different encoder would confound the two.

**What is masked.** Square patches of the image, not random pixels. Masking single pixels is
close to trivial for a convolution -- it interpolates from the four neighbours -- whereas a
whole patch forces the model to use the phenological structure. On an 8x8 image with
`patch=2`, 16 patches, masking 50-75% leaves 4-8 visible.

**On leakage.** Pretraining never sees a target, so no response leaks. It does see pixels of
plots that later fall in a test fold, which is input-side and standard practice for SSL; it
has to be declared. The strict alternative -- pretraining once per fold -- costs 5x and is
worth running only if the result is good enough to defend.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models_conv import PhenoNetS, PhenoNetV


class ConvMAE(nn.Module):
    """`PhenoNetS`/`PhenoNetV` trunk plus a small decoder, trained to inpaint patches.

    The decoder is deliberately weak (two conv layers). A strong decoder lets the encoder
    stay lazy: the reconstruction gets solved downstream and the features never have to
    carry the phenology. It is discarded after pretraining in any case.
    """

    def __init__(self, c_in: int = 1, width: str = "B", arch: str = "sep",
                 pad_mode="zeros", patch: int = 2, mask_ratio: float = 0.6):
        super().__init__()
        w = PhenoNetS.WIDTHS[width]
        base = (PhenoNetS if arch == "sep" else PhenoNetV)
        kw = dict(c_in=c_in, width=w, n_out=1, n_ctx=0, pad_mode=pad_mode, fusion="none")
        self.body = base(**kw) if arch == "sep" else base(arch=arch, **kw)
        self.patch = patch
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Conv2d(w[2], w[1], 3, padding=1), nn.GELU(),
            nn.Conv2d(w[1], c_in, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The trunk's feature map, before pooling. `b3` has stride 2, so this is H/2."""
        b = self.body
        h = b.b2(b.b1(b.stem(x)))
        return b.b4(b.b3(h))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feat = self.encode(x * (1.0 - mask))
        out = self.decoder(feat)
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear",
                                            align_corners=False)
        return out


def patch_mask(shape: tuple[int, int, int, int], patch: int, ratio: float,
               gen: torch.Generator, device) -> torch.Tensor:
    """(B, 1, H, W) mask with whole `patch`x`patch` squares set to 1 (= hidden)."""
    b, _, h, w = shape
    ph, pw = max(1, h // patch), max(1, w // patch)
    n = ph * pw
    k = max(1, int(round(ratio * n)))
    flat = torch.zeros(b, n, device=device)
    for i in range(b):
        idx = torch.randperm(n, generator=gen, device=device)[:k]
        flat[i, idx] = 1.0
    m = flat.view(b, 1, ph, pw)
    return nn.functional.interpolate(m, size=(h, w), mode="nearest")


def pretrain(images: np.ndarray, *, width: str = "B", arch: str = "sep", pad_mode="zeros",
             patch: int = 2, mask_ratio: float = 0.6, epochs: int = 30,
             batch_size: int = 256, lr: float = 1e-3, seed: int = 0,
             device: str | None = None, verbose: bool = True) -> dict:
    """Pretrain on ``images`` (N, C, H, W) and return the trunk's ``state_dict``.

    The loss is computed **on the masked patches only**. Averaging over the whole image lets
    the model score well by copying the visible part, which is not a task.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    x = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32))
    model = ConvMAE(c_in=x.shape[1], width=width, arch=arch, pad_mode=pad_mode,
                    patch=patch, mask_ratio=mask_ratio).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(x)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, seen = 0.0, 0
        for i in range(0, n, batch_size):
            xb = x[perm[i:i + batch_size]].to(device, non_blocking=True)
            m = patch_mask(xb.shape[:1] + (1,) + xb.shape[2:], patch, mask_ratio, gen, device)
            pred = model(xb, m)
            # masked patches only
            loss = (((pred - xb) ** 2) * m).sum() / (m.sum() * xb.shape[1] + 1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(xb)
            seen += len(xb)
        sched.step()
        hist.append(tot / seen)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    epoch {ep + 1:>3}/{epochs}  masked MSE {hist[-1]:.5f}", flush=True)

    # only the trunk travels: stem + the four blocks. The head is task-specific and the
    # decoder is scaffolding.
    keep = {k: v.cpu() for k, v in model.body.state_dict().items()
            if k.split(".")[0] in ("stem", "b1", "b2", "b3", "b4")}
    return {"trunk": keep, "history": hist, "config": dict(
        width=width, arch=arch, patch=patch, mask_ratio=mask_ratio, epochs=epochs,
        n_images=int(n), in_channels=int(x.shape[1]), image_shape=tuple(x.shape[2:]))}


def load_trunk(model: nn.Module, trunk: dict, strict: bool = True) -> int:
    """Copy pretrained trunk weights into a `PhenoNetS`. Returns how many tensors matched.

    Shape mismatches are reported rather than silently skipped: a trunk pretrained at one
    width or channel count that is quietly ignored would look exactly like pretraining that
    did not help.
    """
    own = model.state_dict()
    ok, bad = {}, []
    for k, v in trunk.items():
        if k in own and own[k].shape == v.shape:
            ok[k] = v
        else:
            bad.append(k)
    if bad and strict:
        raise ValueError(
            f"{len(bad)} pretrained tensors do not fit this model, e.g. {bad[:3]}. "
            "The checkpoint was pretrained with a different width, architecture or "
            "channel count.")
    model.load_state_dict({**own, **ok})
    return len(ok)
