from pathlib import Path
import random, numpy as np, torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from dataset import build_loaders
from model import create_model

DATA_ROOT = "data_split"
IMG_SIZE  = 240
BATCH     = 32
WORKERS   = 2
EPOCHS    = 50
LR        = 1e-3
BACKBONE  = "efficientnet_b1"
CKPT_PATH = Path("models/best.pth")
USE_AMP   = True

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

def train():
    set_seed(42)
    device = "cuda"

    train_loader, val_loader, _, class_names = build_loaders(
        root=DATA_ROOT, img_size=IMG_SIZE, batch_size=BATCH, num_workers=WORKERS
    )

    model = create_model(num_classes=len(class_names), backbone=BACKBONE).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    ce  = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device=="cuda"))

    best_val_acc = 0.0
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        running_loss, seen = 0.0, 0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device=="cuda")):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)
            pbar.set_postfix(train_loss=f"{running_loss/seen:.4f}")

        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"[Val] loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": class_names,
                "backbone": BACKBONE,
                "img_size": IMG_SIZE,
            }, CKPT_PATH)
            print(f"  -> saved best to {CKPT_PATH} (acc={best_val_acc:.4f})")

if __name__ == "__main__":
    train()
