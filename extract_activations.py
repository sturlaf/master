from itertools import product
import sys
from PIL import Image
from os import listdir

import numpy as np
import torch
from torchvision import transforms

from lucent.modelzoo import *
from lucent.misc.io import show
import lucent.optvis.objectives as objectives
import lucent.optvis.param as param
import lucent.optvis.render as render
import lucent.optvis.transform as transform
from lucent.misc.channel_reducer import ChannelReducer
from lucent.misc.io import show


def transform_image(input_image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model
    return input_batch


@torch.no_grad()
def get_layer(model, layer, X):
    hook = render.ModuleHook(getattr(model, layer))
    model(X)
    hook.close()
    return hook.features


def get_all_layers(model, layers, X):
    hooks = [render.ModuleHook(getattr(model, layer)) for layer in layers]
    model(X)
    for hook in hooks:
        hook.close()
    return [hook.features for hook in hooks]


def get_activations(model, layers, num, path="data/ILSVRC2015/Data/DET/test/"):
    images = []
    for file in listdir(path)[:num]:
        images.append(Image.open(path + file))
    images = [image for image in images if len(image.getbands()) == 3]
    images = torch.cat(list(map(transform_image, images)))
    activations = get_all_layers(model, layers, images)
    max_activations = [act.max(dim=3)[0].max(dim=2)[0] for act in activations]
    torch.save(max_activations, "activations/max_activations.pt")
    mean_activations = [act.mean(dim=[2, 3]) for act in activations]
    torch.save(mean_activations, "activations/mean_activations.pt")
    return activations


def main():
    layers = [
        "inception3a",
        "inception3b",
        "inception4a",
        "inception4b",
        "inception4c",
        "inception4d",
        "inception4e",
        "inception5a",
        "inception5b",
    ]

    model = torch.hub.load("pytorch/vision:v0.10.0", "googlenet", pretrained=True)
    model.eval()
    get_activations(model, layers, 200)


if __name__ == "__main__":
    main()
