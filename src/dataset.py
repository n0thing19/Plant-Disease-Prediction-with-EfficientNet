from pathlib import Path
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_loaders(root: str, img_size: int, batch_size: int, num_workers: int):
    root = Path(root)
    train_dir = root / "train"
    val_dir   = root / "val"
    test_dir  = root / "test"

    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAdjustSharpness(1.5, p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=tf_train)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=tf_eval)
    test_ds  = datasets.ImageFolder(str(test_dir),  transform=tf_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names
