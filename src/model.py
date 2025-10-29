import timm

def create_model(num_classes: int, backbone: str):
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model
