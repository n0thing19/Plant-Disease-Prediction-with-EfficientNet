import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from src.model import create_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


CKPT_PATH = "checkpoints/best_v2b3_1.pth"
IMG_SIZE  = 300

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

def get_prediction_data(img_path, ckpt_path=CKPT_PATH, img_size=IMG_SIZE, threshold=0.55, enable_gradcam=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["classes"]
    backbone    = ckpt.get("backbone", "tf_efficientnetv2_b3")
    
    model = create_model(len(class_names), backbone=backbone)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()
    
    img_pil = Image.open(img_path).convert("RGB")
    tf = _base_tf(img_size)
    
    views = _tta_views(img_pil)
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
    
    top3 = torch.topk(probs, k=min(3, len(class_names)))
    top3_data = []
    for score, idx in zip(top3.values, top3.indices):
        top3_data.append({"label": class_names[int(idx)], "score": float(score)})

    heatmap_pil = None
    
    if enable_gradcam:
        try:
            target_layers = [model.conv_head] 
            cam = GradCAM(model=model, target_layers=target_layers)
            input_tensor = tf(img_pil).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            img_resized = img_pil.resize((img_size, img_size))
            rgb_img = np.float32(img_resized) / 255
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            heatmap_pil = Image.fromarray(visualization)
            
        except Exception as e:
            print(f"Warning: Gagal membuat Grad-CAM. Error: {e}")

    return {
        "label": pred_label,
        "confidence": pred_conf,
        "is_unknown": pred_conf < threshold, 
        "top3": top3_data,
        "gradcam_image": heatmap_pil
    }

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    print(get_prediction_data(p, enable_gradcam=False))