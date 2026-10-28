"""Ortak eğitim döngüsü (Adım 10–13, §6.1): AdamW (omurga / başlık ayrı öğrenme hızı), 1 epoch ısınma + cosine,
val skoruna göre erken durdurma ve en iyi ağırlıklara dönüş, CUDA'da karma hassasiyet (AMP).
"""
import csv
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .config import RESULTS_DIR


def loader(ds, batch_size, train, workers=None):
    workers = min(8, os.cpu_count() or 1) if workers is None else workers
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train and len(ds) > batch_size,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def param_groups(model, lr_backbone, lr_head):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    return [{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": lr_backbone},
            {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": lr_head}]


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict(model, data_loader, device):
    """(model çıktıları, hedefler) — sıra veri kümesinin sırasıyla aynı (shuffle=False)."""
    model.eval()
    outs, ys = [], []
    for x, y, *_ in data_loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outs.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(outs), torch.cat(ys)


def fit(model, train_loader, val_loader, loss_fn, val_score, *, epochs, lr_backbone, lr_head, weight_decay,
        patience, higher_is_better, device, history_path=None):
    """val_score(model) → tek sayı (sınıflandırma: macro-F1, kalori: MAE). En iyi epoch'un ağırlıklarıyla döner."""
    opt = torch.optim.AdamW(param_groups(model, lr_backbone, lr_head), weight_decay=weight_decay)
    sched = warmup_cosine(opt, len(train_loader), epochs * len(train_loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, best_epoch, bad, history = None, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, *_ in train_loader:
            x, y = x.to(device, non_blocking=True), torch.as_tensor(y).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(x)
            n += len(x)
        score = val_score(model)
        improved = best is None or (score > best if higher_is_better else score < best)
        if improved:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append({"epoch": epoch, "train_loss": round(total / max(n, 1), 5), "val_score": round(score, 5),
                        "lr_head": opt.param_groups[1]["lr"], "sec": round(time.time() - t0, 1)})
        print(f"  epoch {epoch:>2}/{epochs}  train kaybı {total / max(n, 1):.4f}  val {score:.4f}"
              f"{'  ← en iyi' if improved else ''}  ({time.time() - t0:.0f} sn)", flush=True)
        if bad >= patience:
            print(f"  erken durdurma: {patience} epoch boyunca iyileşme yok", flush=True)
            break
    model.load_state_dict(best_state)
    if history_path:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
    return best, best_epoch, history


def history_path(name):
    return str(RESULTS_DIR / "history" / f"{name}.csv")


def ms_per_image(model, device, size=224, batch=32, reps=10):
    """Tablo 1 için çıkarım süresi (yalnızca model; veri okuma hariç)."""
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(3):
            model(x)
        sync = torch.cuda.synchronize if device.type == "cuda" else (torch.mps.synchronize if device.type == "mps" else lambda: None)
        sync()
        t = time.time()
        for _ in range(reps):
            model(x)
        sync()
    return 1000 * (time.time() - t) / (reps * batch)
