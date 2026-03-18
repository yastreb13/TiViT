
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def load_tivit_model(weights_path, device):
    # базовый ViT
    model = models.vit_b_16()
    # делаем 2 выхода
    num_features=model.heads.head.in_features
    model.heads.head=nn.Linear(num_features, 2)
    # загружаем веса с обученной модели
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    # инференнс
    model.eval() 
    return model

# предобработка идем от нампай массива к тензору
transform_frame = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])