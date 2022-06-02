import numpy as np
import os
import pandas as pd
from lucent.optvis import render, objectives
import torch


def make_images(layer, num_clusters=5, buckets=40, save_location="images/"):
    df = pd.read_pickle(f"data/perera/{layer}.pkl")
    df = df.head(num_clusters)
    activity = np.load(f"activations/ILSVRC2015/{layer}.npy")
    num_of_neurons = activity.shape[1]
    buckets_original = buckets
    for index, row in df.iterrows():
        buckets = buckets_original
        cluster = activity[row["cluster_members"]]
        f = row["circle_param"]
        clusters_overlap = [[]]
        while 0 in list(map(len, clusters_overlap)):
            print(f"Layer: {layer}. Cluster: {index}. Buckets: {buckets}")
            linsp = np.linspace(f.min(), f.max(), buckets + 1)

            clusters_overlap = []
            for a in range(1, buckets):
                c = []
                for p in range(len(f)):
                    if linsp[a - 1] < f[p] < linsp[a + 1]:
                        c.append(cluster[p])
                clusters_overlap.append(c)

            c = []
            for p in range(len(f)):
                if linsp[buckets - 1] < f[p] or f[p] < linsp[1]:
                    c.append(cluster[p])
            clusters_overlap.append(c)
            av_clusters_overlap = [
                np.mean(overlap, axis=0) for overlap in clusters_overlap
            ]
            buckets = buckets // 2

            # print(list(map(lambda a: a.shape, av_clusters_overlap)))
            # print(list(map(len, clusters_overlap)))

        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "googlenet", pretrained=True
        ).eval()

        pics = []
        channel = lambda n: objectives.channel(layer, n)
        high_inform = list(
            range(num_of_neurons)
        )  # Can be just a few of the neurons by using the information rates
        for n in range(len(av_clusters_overlap)):
            a = []
            for m in range(len(high_inform)):
                a.append(av_clusters_overlap[n][m] * channel(high_inform[m]))
            obj = sum(a)
            d = render.render_vis(model, obj, show_image=False, thresholds=[256])
            pics.append(d[0][0])
        np.save(f"{save_location}{layer}/cluster_{index}_", np.array(pics))


def main():
    layers = [
        "inception3a",
        # "inception3b",
        # "inception4a",
        # "inception4b",
        # "inception4c",
        # "inception4d",
        # "inception4e",
        # "inception5a",
        # "inception5b",
    ]
    save_location = "images/"
    make_save_locations(layers=layers, save_location=save_location)
    for layer in layers:
        print(layer)
        make_images(layer=layer, num_clusters=1, buckets=2, save_location="images/")


def make_save_locations(layers, save_location="images/"):
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    for layer in layers:
        loc = f"{save_location}{layer}/"
        if not os.path.exists(loc):
            os.makedirs(loc)


def test():
    circle_images = np.load("images/inception3a/cluster_8_.npy", "r")
    print(circle_images.shape)
    from PIL import Image

    a = circle_images[1]

    a = (a * 255).astype(np.uint8)

    ima = Image.fromarray(obj=a, mode="RGB")
    ima.show()


if __name__ == "__main__":
    main()
    # test()
