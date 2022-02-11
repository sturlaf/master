import os
import random

import lucent.optvis.render as render
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


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


def select_patch(image_activation, patch_size):
    x_cor, y_cor = random.randint(0, patch_size), random.randint(0, patch_size)
    return image_activation[:, :, x_cor, y_cor]


def get_activations(
    model,
    layers,
    batch_size=150,
    number_of_images=100000,
    path="data/ILSVRC2015/Data/DET/test/",
):
    for layer in layers:
        if not os.path.exists(f"activations/{layer}"):
            os.makedirs(f"activations/{layer}")
    image_paths = os.listdir(path)[:number_of_images]
    total_num_pictures = len(image_paths)
    pbar = tqdm(total=total_num_pictures)
    while image_paths:
        batch_paths, image_paths = image_paths[:batch_size], image_paths[batch_size:]
        images = []
        for file in batch_paths:
            image = Image.open(path + file)
            # Only want pictures with 3 image chanels, i.e colored pictures
            if len(image.getbands()) == 3:
                images.append(transform_image(image))
            image.close()
        images = torch.cat(images)
        activations = get_all_layers(model, layers, images)
        for layer, layer_activations in zip(layers, activations):
            patch_size = layer_activations.shape[3] - 1
            layer_activations = map(
                lambda img_activation: select_patch(img_activation, patch_size),
                layer_activations.split(1),
            )
            layer_activations = torch.cat(list(layer_activations))
            torch.save(
                layer_activations,
                f"activations/{layer}/activations_{total_num_pictures - len(image_paths)}.pt",
            )
        pbar.update(batch_size)


def main():
    random.seed(17)
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

    extact_activations(layers)
    concat_batches(layers)


def extact_activations(layers):
    model = torch.hub.load("pytorch/vision:v0.10.0", "googlenet", pretrained=True)
    model.eval()
    get_activations(model, layers)


def concat_batches(
    layers, folder="activations", save_location="activations/ILSVRC2015"
):
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    for layer in layers:
        batch_paths = os.listdir(f"{folder}/{layer}")
        activations = torch.cat(
            [torch.load(f"{folder}/{layer}/{file}") for file in batch_paths]
        )
        torch.save(activations, f"{save_location}/{layer}.pt")


if __name__ == "__main__":
    main()
