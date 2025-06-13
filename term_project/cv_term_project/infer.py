import torch
import json
import numpy as np
from argparse import ArgumentParser
from torchvision.transforms import transforms
from PIL import Image
from matplotlib import pyplot as plt

from model.resnet_50 import ResNet50


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def infer(model, image_path, weight_path):
    image = Image.open(image_path).convert('RGB')
    image = val_transform(image)

    model_state_dict = torch.load(weight_path)['model_state_dict']

    model.load_state_dict(model_state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output: np.ndarray = model(image).squeeze().cpu().numpy()

    return output.argmax(axis=-1)

def display_res(image, cls):
    plt.imshow(image)
    plt.title(cls)
    plt.axis('off')
    plt.show()

def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None
    )
    parser.add_argument(
        "--weight",
        default=None
    )
    parser.add_argument(
        "--image",
        default=None
    )

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset:
        with open(args.dataset, 'r') as fp:
            dataset_json = json.load(fp)
        classes = dataset_json['classes']

        model = ResNet50(len(classes))
        index = infer(model, args.image, args.weight)
        image = Image.open(args.image).convert('RGB')
        
        display_res(image, classes[index])

    else:
        print('--dataset must be set.')