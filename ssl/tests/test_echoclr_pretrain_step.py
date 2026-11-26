"""Smoke test: one EchoCLR pretraining step on synthetic echo videos (CPU).

Mirrors the EchoCLR branch of ssl/main.py (lines 55-78), but builds the dataset
with multi_instance=True and frame_reordering=True directly, because main.py as
committed does not pass those flags through (and hard-codes cuda:0).

Run from the repo root:  pytest ssl/tests
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torchvision

SSL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SSL_DIR)

from dataset import EchoDataset  # noqa: E402
from losses import NT_Xent  # noqa: E402
from model import SimCLR  # noqa: E402
from utils import set_seed  # noqa: E402

# 4! = 24 orderings, matching the hard-coded Linear(n_features, 24) reordering head
CLIP_LEN = 4
N_FRAMES = 12
SIZE = 64
BATCH_SIZE = 4
# Videos per study -> 3 + 1 + 1 = 5 positive pairs (single-video study is paired with itself)
STUDIES = {1001: 3, 1002: 2, 1003: 1}


def _write_avi(path, n_frames, size, rng):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (size, size))
    assert writer.isOpened(), f"Could not open VideoWriter for {path}"
    for _ in range(n_frames):
        writer.write(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    writer.release()


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(7193)
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    rows = []
    for acc_num, n_videos in STUDIES.items():
        for video_num in range(n_videos):
            fname = f"{acc_num}_{video_num}.avi"
            _write_avi(str(video_dir / fname), N_FRAMES, SIZE, rng)
            rows.append(
                {
                    "plax_prob": 0.9,
                    "fpath": fname,
                    "acc_num": acc_num,
                    "av_stenosis": "None",
                    "video_num": video_num,
                }
            )

    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    return str(tmp_path)


def _echoclr_dataset(data_dir):
    return EchoDataset(
        data_dir=data_dir,
        split="train",
        clip_len=CLIP_LEN,
        sampling_rate=1,
        multi_instance=True,
        frame_reordering=True,
    )


def test_positive_pairs_come_from_same_study(data_dir):
    ds = _echoclr_dataset(data_dir)
    study_of = dict(zip(ds.label_df["fpath"], ds.label_df["acc_num"]))

    assert len(ds) == 5
    assert len(ds.temporal_orderings) == 24
    for f_i, f_j in zip(ds.fnames_i, ds.fnames_j):
        assert study_of[f_i] == study_of[f_j]
        # Only the single-video study should be paired with itself
        assert (f_i == f_j) == (study_of[f_i] == 1003)


def test_reordering_label_matches_applied_permutation(data_dir):
    ds = _echoclr_dataset(data_dir)

    # Frame k is a constant image of brightness 40*(k+1); pad/crop, flip, and rotation
    # are shared across frames, so per-frame mean brightness keeps the original order.
    clip = np.stack(
        [
            np.full((SIZE, SIZE, 3), 40 * (k + 1), dtype=np.uint8)
            for k in range(CLIP_LEN)
        ]
    )

    for seed in range(20):
        set_seed(seed)
        x, label = ds._augment(clip)
        frame_means = x.reshape(CLIP_LEN, -1).mean(axis=1)
        original_index = np.argsort(np.argsort(frame_means))
        np.testing.assert_array_equal(original_index, ds.temporal_orderings[label])


def test_one_echoclr_pretraining_step(data_dir):
    set_seed(2716)
    ds = _echoclr_dataset(data_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
    )
    x_i, x_j, t_i, t_j = next(iter(loader))

    # Data: (B, C, T, H, W) clips in [0, 1], integer permutation labels in [0, 24)
    for x in (x_i, x_j):
        assert x.shape == (BATCH_SIZE, 3, CLIP_LEN, SIZE, SIZE)
        assert x.dtype == torch.float32
        assert torch.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    for t in (t_i, t_j):
        assert t.shape == (BATCH_SIZE,) and t.dtype == torch.int64
        assert ((t >= 0) & (t < 24)).all()

    encoder = torchvision.models.video.r3d_18()  # random init, as in ssl/main.py:36
    model = SimCLR(
        encoder=encoder,
        projection_dim=128,
        n_features=encoder.fc.in_features,
        frame_reordering=True,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fxn = NT_Xent(BATCH_SIZE, temperature=0.05, world_size=1)
    cls_loss_fxn = torch.nn.CrossEntropyLoss()

    h_i, h_j, z_i, z_j, t_hat_i, t_hat_j = model(x_i, x_j)
    assert h_i.shape == h_j.shape == (BATCH_SIZE, 512)
    assert z_i.shape == z_j.shape == (BATCH_SIZE, 128)
    assert t_hat_i.shape == t_hat_j.shape == (BATCH_SIZE, 24)

    contrastive_loss = loss_fxn(z_i, z_j)
    reordering_loss = cls_loss_fxn(torch.cat([t_hat_i, t_hat_j]), torch.cat([t_i, t_j]))
    loss = contrastive_loss + reordering_loss
    assert torch.isfinite(loss) and contrastive_loss > 0 and reordering_loss > 0

    tracked = [
        "encoder.stem.0.weight",
        "encoder.layer4.1.conv2.0.weight",
        "projector.0.weight",
        "reordering_head.weight",
    ]
    params = dict(model.named_parameters())
    before = {name: params[name].detach().clone() for name in tracked}

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name in tracked:
        assert params[name].grad is not None and params[name].grad.abs().sum() > 0, name
    optimizer.step()

    for name in tracked:
        assert not torch.equal(before[name], params[name].detach()), (
            f"{name} was not updated"
        )
