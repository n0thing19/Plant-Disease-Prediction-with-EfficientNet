from pathlib import Path
import shutil
import random

SRC_PATH = Path("../.cache_kagglehub/datasets/nirmalsankalana/plant-diseases-training-dataset/versions/12/data")
RAW_DIR  = Path("../data_raw")
OUT_DIR  = Path("../data_split")
SUB_OUT_DIRS = ["train", "val", "test"]

print("[1/3] Memastikan direktori telah dibuat...")
if not RAW_DIR.exists():
    RAW_DIR.mkdir()
if not OUT_DIR.exists():
    OUT_DIR.mkdir()

for sub_dir in SUB_OUT_DIRS:
    sub_out_path = OUT_DIR / sub_dir
    sub_out_path.mkdir(parents=True, exist_ok=True)
print("Selesai membuat direktori.")       

print("\n[2/3] Menyalin folder 'Apple...' dari SRC_PATH ke RAW_DIR/ ...")     

for item in SRC_PATH.iterdir():
    if item.is_dir() and item.name.startswith("Apple"):
        dst = RAW_DIR / item.name
        if not dst.exists():
            shutil.copytree(item, dst)
else:
    print("Selesai menyalin data kelas Apple ke", RAW_DIR)

print("\n[3/3] Membagi data train/val/test sesuai rasio...")

random.seed(42)
for item in RAW_DIR.iterdir():
    if item.is_dir() and item.name.startswith("Apple"):
        images = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"]
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * 0.8)
        n_val   = int(n_total * 0.1)
        n_test  = n_total - n_train - n_val

        splits={
            "train": images[:n_train],
            "val":   images[n_train:n_train+n_val],
            "test":  images[n_train+n_val:]
        }

        for img, split_imgs in splits.items():
            dst = OUT_DIR / img / item.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dst / img.name)
        
else:
    print("Selesai membagi data ke dalam", OUT_DIR)