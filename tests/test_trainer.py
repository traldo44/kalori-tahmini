import torch
from torch.utils.data import TensorDataset

from src.models import build_model, decode_target, encode_target, load_checkpoint, model_norm, save_checkpoint
from src.trainer import fit, loader, predict, warmup_cosine


def test_encode_decode_with_scale():
    y = torch.tensor([[250.0, 180.0], [40.0, 60.0]])
    for target in ("raw", "log1p"):
        scale = encode_target(y, target).mean(0)
        assert torch.allclose(encode_target(y, target, scale).mean(0), torch.ones(2), atol=1e-5)
        assert torch.allclose(decode_target(encode_target(y, target, scale), target, scale), y, atol=1e-2)


def test_warmup_then_cosine_to_zero():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=4, total_steps=20)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    assert lrs[0] == 0.25 and max(lrs) == 1.0 and lrs[-1] < 0.01


def test_fit_runs_and_restores_best_weights(tmp_path):
    torch.manual_seed(0)
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 3, (16,))
    dl = loader(TensorDataset(x, y), batch_size=8, train=True, workers=0)
    val = loader(TensorDataset(x, y), batch_size=8, train=False, workers=0)
    model = build_model("resnet18", n_outputs=3, pretrained=False)
    scores = iter([0.5, 0.9, 0.1])  # en iyi 2. epoch
    snapshots = []
    def score(m):
        snapshots.append({k: v.clone() for k, v in m.state_dict().items()})
        return next(scores)
    best, best_epoch, hist = fit(model, dl, val, torch.nn.CrossEntropyLoss(), score, epochs=3, lr_backbone=1e-3,
                                 lr_head=1e-2, weight_decay=0.0, patience=5, higher_is_better=True,
                                 device=torch.device("cpu"), history_path=str(tmp_path / "h.csv"))
    assert (best, best_epoch, len(hist)) == (0.9, 2, 3)
    assert all(torch.equal(v, snapshots[1][k]) for k, v in model.state_dict().items())  # 2. epoch'a dönüldü
    assert (tmp_path / "h.csv").read_text().count("\n") == 4
    out, ys = predict(model, val, torch.device("cpu"))
    assert out.shape == (16, 3) and torch.equal(ys, y)


def test_checkpoint_keeps_scale_and_normalization(tmp_path):
    model = build_model("resnet18", n_outputs=2, pretrained=False)
    save_checkpoint(tmp_path / "m.pt", model, "resnet18", outputs=("calories", "mass"), scale=torch.tensor([250.0, 200.0]),
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), test_MAE=80.0)
    _, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["scale"] == [250.0, 200.0] and meta["mean"] == [0.5, 0.5, 0.5] and meta["info"]["test_MAE"] == 80.0


def test_vit_uses_its_own_normalization():
    assert model_norm(build_model("vit_tiny_patch16_224", 1, pretrained=False))[0] == (0.5, 0.5, 0.5)
