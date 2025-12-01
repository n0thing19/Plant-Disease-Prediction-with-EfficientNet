from pathlib import Path
import random, numpy as np, torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from dataset import build_loaders
from model import create_model

DATA_ROOT = "data_final"         
IMG_SIZE  = 300
BATCH     = 24
WORKERS   = 2
EPOCHS    = 16
LR_HEAD   = 1e-3
LR_ALL    = 1e-4                 
BACKBONE  = "tf_efficientnetv2_b3"
CKPT_PATH = Path("checkpoints/best_v2b3_1.pth")
USE_AMP   = True

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = ce(out, y)
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

def train():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, _, class_names = build_loaders(
        root=DATA_ROOT, img_size=IMG_SIZE, batch_size=BATCH, num_workers=WORKERS
    )

    model = create_model(len(class_names), backbone=BACKBONE).to(device)
    ce = nn.CrossEntropyLoss()

    print("\nTahap 1: Training classifier (frozen backbone)...\n")

    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "get_classifier"):
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    for epoch in range(1, 6): 
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/5", ncols=100)
        running_loss, count = 0.0, 0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and device == "cuda")):
                out = model(x)
                loss = ce(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            count += 1
            pbar.set_postfix(train_loss=f"{running_loss/count:.4f}")

        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"[Val] loss={val_loss:.4f} acc={val_acc:.4f}")

    print("\nTahap 2: Fine-tuning seluruh model EfficientNetV2-B3...\n")

    for param in model.parameters():
        param.requires_grad = True

    opt = AdamW(model.parameters(), lr=LR_ALL)
    best_val_acc = 0.0

    for epoch in range(6, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
        running_loss, count = 0.0, 0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and device == "cuda")):
                out = model(x)
                loss = ce(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            count += 1
            pbar.set_postfix(train_loss=f"{running_loss/count:.4f}")

        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"[Val] loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "classes": class_names,
                "backbone": BACKBONE,
                "img_size": IMG_SIZE,
            }, CKPT_PATH)
            print(f"  Model terbaik disimpan ke {CKPT_PATH} (acc={best_val_acc:.4f})")

    print(f"\nTraining selesai. Model terbaik akurasi: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
