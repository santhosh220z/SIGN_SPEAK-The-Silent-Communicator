import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import SignTransformer, PreprocessLayer
from train import triplet_loss, contrastive_loss


def test_preprocess_output_shapes():
    pre = PreprocessLayer(max_len=64)
    x = torch.randn(2, 64, 543, 3)
    feats = pre(x)
    assert feats['appearance'].shape == (2, 64, pre.app_dim)
    assert feats['motion'].shape == (2, 64, pre.mot_dim)
    assert not torch.isnan(feats['appearance']).any()
    assert not torch.isnan(feats['motion']).any()


def test_preprocess_handles_all_nan_frame():
    pre = PreprocessLayer(max_len=64)
    x = torch.full((1, 64, 543, 3), float('nan'))
    feats = pre(x)
    assert not torch.isnan(feats['appearance']).any()
    assert not torch.isnan(feats['motion']).any()


def test_sign_transformer_forward_and_embedding():
    model = SignTransformer(num_classes=10, dim=192, max_len=64)
    x = torch.randn(2, 64, 543, 3)
    logits = model(x)
    assert logits.shape == (2, 10)
    emb = model.get_embedding(x)
    assert emb.shape == (2, 128)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_triplet_loss_zero_for_well_separated_clusters():
    # Two tight, far-apart clusters: hardest_pos ~ 0, hardest_neg large -> 0 loss
    a = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
    b = torch.tensor([[5.0, 5.0], [5.01, 5.0], [5.0, 5.01]])
    emb = torch.cat([a, b], dim=0)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    assert triplet_loss(emb, labels, margin=0.3).item() == 0.0


def test_triplet_loss_positive_when_clusters_overlap():
    emb = torch.tensor([[0.0, 0.0], [0.05, 0.0], [0.02, 0.0], [0.03, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert triplet_loss(emb, labels, margin=0.5).item() > 0.0


def test_contrastive_loss_prefers_same_class_closer():
    same_close = torch.tensor([[0.0, 0.0], [0.01, 0.0], [3.0, 3.0], [3.01, 3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    good = contrastive_loss(same_close, labels)
    bad_emb = torch.tensor([[0.0, 0.0], [2.0, 2.0], [0.05, 0.0], [2.05, 2.0]])
    bad = contrastive_loss(bad_emb, labels)
    assert good.item() < bad.item()
