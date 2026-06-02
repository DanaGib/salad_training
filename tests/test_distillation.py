"""Unit tests for the JEPA cross-modal distillation components.

All tests run on CPU with random tensors. The HuggingFace download is
intercepted by a lightweight monkeypatch so no network or GPU is required.

Run from the repo root:
    pytest tests/test_distillation.py -v
"""
import sys
import types
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, '.')


# ---------------------------------------------------------------------------
# 1. local_distill_loss: output is a scalar tensor
# ---------------------------------------------------------------------------
def test_distill_loss_returns_scalar():
    from utils.distill_loss import local_distill_loss
    pred   = torch.randn(4, 256, 768)
    target = torch.randn(4, 256, 768)
    loss = local_distill_loss(pred, target)
    assert loss.shape == (), f"loss must be scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 2. local_distill_loss: gradient flows to pred but not to target
# ---------------------------------------------------------------------------
def test_distill_loss_gradient_flows():
    from utils.distill_loss import local_distill_loss
    pred   = torch.randn(2, 256, 768, requires_grad=True)
    target = torch.randn(2, 256, 768)
    loss = local_distill_loss(pred, target)
    loss.backward()
    assert pred.grad is not None, "gradient must reach pred"
    assert target.grad is None, "target must not accumulate gradients"


# ---------------------------------------------------------------------------
# 3. local_distill_loss: zero loss when inputs are already unit-normalized
#    and identical (cosine-direction MSE = 0)
# ---------------------------------------------------------------------------
def test_distill_loss_zero_on_identical():
    import torch.nn.functional as F
    from utils.distill_loss import local_distill_loss
    x = torch.randn(2, 256, 768)
    x_norm = F.normalize(x, p=2, dim=-1)
    loss = local_distill_loss(x_norm, x_norm)
    assert loss.item() < 1e-6, f"loss must be ~0 on identical inputs, got {loss.item()}"


# ---------------------------------------------------------------------------
# 4. SALAD.project_patches: output shape [B, num_patches, depth_dim]
# ---------------------------------------------------------------------------
def test_salad_project_patches_shape():
    from models.aggregators.salad import SALAD
    salad = SALAD(num_channels=768, depth_anything_hidden_dim=768)
    salad.eval()
    feat_map = torch.randn(2, 768, 16, 16)
    out = salad.project_patches(feat_map)
    assert out.shape == (2, 256, 768), f"expected (2,256,768), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. SALAD.forward: still returns only the global descriptor (no API change)
# ---------------------------------------------------------------------------
def test_salad_forward_returns_descriptor_only():
    from models.aggregators.salad import SALAD
    salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
    salad.eval()
    feat_map  = torch.randn(2, 768, 16, 16)
    cls_token = torch.randn(2, 768)
    out = salad((feat_map, cls_token))
    expected_dim = 256 + 64 * 128
    assert out.shape == (2, expected_dim), f"expected (2,{expected_dim}), got {out.shape}"
    assert not isinstance(out, tuple), "forward must return a tensor, not a tuple"


# ---------------------------------------------------------------------------
# 6. DepthTeacher: all parameters have requires_grad=False
# ---------------------------------------------------------------------------
def test_teacher_parameters_frozen(monkeypatch):
    _inject_mock_depth_anything(monkeypatch)
    from models.teacher import DepthTeacher
    teacher = DepthTeacher()
    trainable = [p for p in teacher.parameters() if p.requires_grad]
    assert len(trainable) == 0, "teacher must have no trainable parameters"


# ---------------------------------------------------------------------------
# 7. DepthTeacher: stays in eval mode even after Lightning calls .train()
# ---------------------------------------------------------------------------
def test_teacher_stays_eval_after_train_call(monkeypatch):
    _inject_mock_depth_anything(monkeypatch)
    from models.teacher import DepthTeacher
    teacher = DepthTeacher()
    teacher.train()
    assert not teacher.training, "teacher must stay in eval mode after .train() call"


# ---------------------------------------------------------------------------
# 8. DepthTeacher forward: output shape [B, num_patches, 768]
# ---------------------------------------------------------------------------
def test_teacher_output_shape(monkeypatch):
    _inject_mock_depth_anything(monkeypatch)
    from models.teacher import DepthTeacher
    teacher = DepthTeacher()
    images = torch.randn(2, 3, 224, 224)
    out = teacher(images)
    assert out.shape == (2, 256, 768), f"expected (2,256,768), got {out.shape}"


# ---------------------------------------------------------------------------
# Mock helper: replaces AutoModelForDepthEstimation with a CPU stub so tests
# run without downloading the 300 MB checkpoint or requiring a GPU.
# ---------------------------------------------------------------------------
def _inject_mock_depth_anything(monkeypatch):
    """Patch sys.modules so DepthTeacher loads a tiny fake encoder."""

    class _FakeEncoder(nn.Module):
        """Mimics Dinov2Model: returns last_hidden_state [B, num_patches+1, 768]."""
        def forward(self, pixel_values, output_hidden_states=False):
            B = pixel_values.shape[0]
            out = types.SimpleNamespace()
            out.last_hidden_state = torch.zeros(B, 257, 768)
            return out

    class _FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _FakeEncoder()

    class _FakeDepthModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _FakeBackbone()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name):
            return _FakeDepthModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForDepthEstimation = _FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # force DepthTeacher to reload with the patched transformers module
    sys.modules.pop("models.teacher", None)
