# src/eval.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import build_loaders
from model import create_model

CKPT = "models/best.pth"

def evaluate():
    ckpt = torch.load(CKPT, map_location="cpu")
    class_names = ckpt["classes"]
    img_size    = ckpt.get("img_size", 240)
    backbone    = ckpt.get("backbone", "efficientnet_b1")

    model = create_model(len(class_names), backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    _, _, test_loader, _ = build_loaders(img_size=img_size)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            y_true += y.tolist()
            y_pred += out.argmax(1).cpu().tolist()

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
