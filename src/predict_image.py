import torch, os
from torchvision import transforms
from PIL import Image
from model import create_model

CKPT_PATH = "models/best.pth"
IMG_PATH  = "detail-powdery-mildew-plant-disease-close-up-395151004.jpg"  
IMG_SIZE  = 224
UNKNOWN_THRESHOLD = 0.55    

def _base_tf(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def _tta_views(img):
    return [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.rotate(90, expand=True),
    ]

def predict_image(img_path=IMG_PATH, ckpt_path=CKPT_PATH, img_size=IMG_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["classes"]
    backbone    = ckpt.get("backbone", "efficientnet_b0")
    img_size    = ckpt.get("img_size", img_size)

    print(f"Ditemukan {len(class_names)} kelas dari folder training: {', '.join(class_names)}")
    print("Loading model structure...")
    model = create_model(len(class_names), backbone=backbone)
    print(f"Loading trained weights from '{os.path.basename(ckpt_path)}'...\n")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()

    tf = _base_tf(img_size)
    img = Image.open(img_path).convert("RGB")
    print(f"Memprediksi gambar: {os.path.basename(img_path)}\n")

    views = _tta_views(img)
    with torch.no_grad():
        probs_accum = torch.zeros(len(class_names), device=device)
        for v in views:
            x = tf(v).unsqueeze(0).to(device)
            out = model(x)
            probs_accum += torch.softmax(out, dim=1)[0]
        probs = (probs_accum / len(views)).cpu()

    pred_idx = int(probs.argmax().item())
    pred_conf = float(probs[pred_idx].item())
    pred_label = class_names[pred_idx]

    print("="*30)
    print("      HASIL PREDIKSI")
    print("="*30)
    if pred_conf < UNKNOWN_THRESHOLD:
        print(f"\nPrediksi Kelas: (Unknown/Ragu) → kandidat: {pred_label}")
        print(f"Tingkat Keyakinan (Confidence): {pred_conf*100:.2f}%\n")
    else:
        print(f"\nPrediksi Kelas: {pred_label}")
        print(f"Tingkat Keyakinan (Confidence): {pred_conf*100:.2f}%\n")

    print("Probabilitas semua kelas:")
    for i, cls in enumerate(class_names):
        print(f"  - {cls}: {probs[i].item()*100:.2f}%")

    top3 = torch.topk(probs, k=min(3, len(class_names)))
    print("\nTop-3 kandidat:")
    for score, idx in zip(top3.values, top3.indices):
        print(f"  * {class_names[int(idx)]}: {float(score)*100:.2f}%")

    return pred_label, pred_conf

if __name__ == "__main__":
    predict_image()
