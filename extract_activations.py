import datetime
from itertools import product
from os import listdir

import lucent.optvis.render as render
import torch
from PIL import Image
from torchvision import transforms


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


def get_activations(model, layers, num=100000, path="data/ILSVRC2015/Data/DET/test/"):
    images = []
    for file in listdir(path)[:num]:
        images.append(Image.open(path + file))
    images = [image for image in images if len(image.getbands()) == 3]
    max_activations = []
    mean_activations = []
    while images:
        print(f"{len(images)}", end=" - ")
        batch = images[:500]
        images = images[500:]
        batch = torch.cat(list(map(transform_image, batch)))
        activations = get_all_layers(model, layers, batch)
        max_activations = [
            torch.cat((max_act, act.max(dim=3)[0].max(dim=2)[0]))
            for max_act, act in zip(max_activations, activations)
        ]
        mean_activations = [
            torch.cat((mean_act, act.max(dim=3)[0].max(dim=2)[0]))
            for mean_act, act in zip(mean_activations, activations)
        ]
    torch.save(max_activations, "activations/max_activations.pt")
    torch.save(mean_activations, "activations/mean_activations.pt")
    return activations


def main():
    t = datetime.datetime.now()
    print(f"Started at {t}")
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
    get_activations(model, layers)
    print(f"Finished in {datetime.datetime.now() - t}")


if __name__ == "__main__":
    main()
