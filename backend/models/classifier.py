import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

_loaded_models = {}

MODEL_CONFIG = {
    'brain_tumor': {
        'file': 'brain_tumor_efficientnet_b3.pth',
        'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'display': {
            'glioma': 'Glioma Tumor',
            'meningioma': 'Meningioma Tumor',
            'notumor': 'No Tumor Detected',
            'pituitary': 'Pituitary Tumor',
        },
    },
    'chest_xray': {
        'file': 'chest_xray_efficientnet_b3.pth',
        'classes': ['images', 'sample'],
        'display': {
            'images': 'Pneumonia Detected',
            'sample': 'Normal',
        },
    },
    'covid19': {
        'file': 'covid19_efficientnet_b3.pth',
        'classes': ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'],
        'display': {
            'COVID': 'COVID-19 Positive',
            'Lung_Opacity': 'Lung Opacity Detected',
            'Normal': 'Normal',
            'Viral Pneumonia': 'Viral Pneumonia',
        },
    },
    'malaria': {
        'file': 'malaria_efficientnet_b3.pth',
        'classes': ['Parasitized', 'Uninfected', 'cell_images'],
        'display': {
            'Parasitized': 'Malaria Parasitized',
            'Uninfected': 'Uninfected (Normal)',
            'cell_images': 'Cell Images',
        },
    },
    'fracture': {
        'file': 'fracture_efficientnet_b3.pth',
        'classes': ['test', 'train', 'val'],
        'display': {
            'test': 'Fracture Detected',
            'train': 'No Fracture',
            'val': 'Indeterminate',
        },
    },
}

_preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_model(model_key: str):
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    import timm

    config = MODEL_CONFIG[model_key]
    path = os.path.join(MODELS_DIR, config['file'])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    num_classes = checkpoint.get('num_classes', len(config['classes']))
    ckpt_sd = checkpoint['model_state_dict']

    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
    model_sd = model.state_dict()

    filtered = {}
    for k in model_sd:
        if k in ckpt_sd and model_sd[k].shape == ckpt_sd[k].shape:
            filtered[k] = ckpt_sd[k]
        else:
            filtered[k] = model_sd[k]

    model.load_state_dict(filtered, strict=False)
    model.eval()

    _loaded_models[model_key] = (model, config)
    return model, config


def classify_image(image_bytes: bytes, model_key: str) -> dict:
    model, config = _load_model(model_key)

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = _preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_idx = torch.argmax(probs).item()
    top_class = config['classes'][top_idx]
    top_label = config['display'].get(top_class, top_class)
    confidence = probs[top_idx].item()

    all_results = []
    for i, cls in enumerate(config['classes']):
        all_results.append({
            'class': cls,
            'label': config['display'].get(cls, cls),
            'confidence': round(probs[i].item() * 100, 2),
        })

    all_results.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        'model': model_key,
        'prediction': top_label,
        'prediction_class': top_class,
        'confidence': round(confidence * 100, 2),
        'all_results': all_results,
    }


def get_available_models():
    available = []
    for key, config in MODEL_CONFIG.items():
        path = os.path.join(MODELS_DIR, config['file'])
        available.append({
            'key': key,
            'name': key.replace('_', ' ').title(),
            'classes': list(config['display'].values()),
            'available': os.path.exists(path),
        })
    return available
