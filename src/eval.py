import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from dataset import build_loaders
from model import create_model

CKPT_PATH = "checkpoints/best_v2b3.pth" 
DATA_ROOT = "data_final"            
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_visual():
    print(f"Evaluasi model dari checkpoint: {CKPT_PATH}\n")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    class_names = ckpt["classes"]
    backbone = ckpt.get("backbone", "tf_efficientnetv2_b3")
    img_size = ckpt.get("img_size", 300)

    _, _, test_loader, _ = build_loaders(
        root=DATA_ROOT, img_size=img_size, batch_size=BATCH_SIZE, num_workers=2
    )

    model = create_model(len(class_names), backbone=backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            preds = out.argmax(1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(preds)

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    acc = report["accuracy"] * 100
    print(f"Overall Accuracy: {acc:.2f}%")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2f}%)", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    per_class_acc = [report[c]["recall"] * 100 for c in class_names]
    sns.barplot(x=class_names, y=per_class_acc, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Recall (%)")
    plt.title("Akurasi per Kelas (Recall)")
    plt.tight_layout()
    plt.show()

    print("\nEvaluasi visual selesai!\n")


if __name__ == "__main__":
    evaluate_visual()
