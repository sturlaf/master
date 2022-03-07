import numpy as np
import os
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from umap import UMAP
from fix_umap_bug import fix_umap_bug
import pandas as pd
from tqdm import tqdm
from persim import plot_diagrams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from circular_cords import get_coords
import plotly.graph_objects as go


def plot_persistance_clusters(layer, num_clusters=5, file_name="persistance_clusters"):
    df = pd.read_pickle(f"data/clusters/{layer}.pkl")
    df = df.head(num_clusters)  # Top 5 longest bars
    activity = np.load(f"activations/ILSVRC2015/{layer}.npy")
    num_of_neurons = activity.shape[1]
    with PdfPages(f"pdf_files/{file_name}_{layer}.pdf") as pdf:
        for index, row in df.iterrows():
            cluster = activity[row["cluster_members"]]
            layout = UMAP(
                n_components=num_of_neurons,
                verbose=True,
                n_neighbors=20,
                min_dist=0.01,
                metric="cosine",
            ).fit_transform(cluster)
            distance = squareform(pdist(layout, "euclidean"))
            coeff = 47
            persistence = ripser(
                X=distance,
                maxdim=1,
                coeff=coeff,
                do_cocycles=True,
                distance_matrix=True,
                thresh=np.max(distance[~np.isinf(distance)]),
            )
            fig, ax = plt.subplots(figsize=(12, 12))
            plot_diagrams(
                diagrams=persistence["dgms"],
                plot_only=None,
                title=f"Cluster nr. {row['cluster_id']}",
                xy_range=None,
                labels=None,
                colormap="default",
                size=20,
                ax_color=np.array([0.0, 0.0, 0.0]),
                diagonal=True,
                lifetime=False,
                legend=True,
                show=False,
                ax=ax,
            )
            pdf.savefig(fig)
            make_3d_figure(
                diagrams=persistence["dgms"][1],
                cocycles=persistence["cocycles"][1],
                layout=layout,
                distance=distance,
                coeff=coeff,
                row=row,
                layer=layer,
                file_name=file_name,
            )


def make_3d_figure(diagrams, cocycles, layout, distance, coeff, row, layer, file_name):
    births1, deaths1 = diagrams[:, 0], diagrams[:, 1]
    lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    threshold = births1[iMax[-1]] + (deaths1[iMax[-1]] - births1[iMax[-1]]) * (9 / 10)
    f, theta_matrix, verts, num_verts = get_coords(
        cocycle=cocycles[iMax[-1]],
        threshold=threshold,
        num_sampled=row["cluster_size"],
        dists=distance,
        coeff=coeff,
        bool_smooth_circle="perea",  # "graph", "old"
    )
    layout_3d = UMAP(
        n_components=3,
        verbose=True,
        n_neighbors=20,
        min_dist=0.01,
        metric="euclidean",
    ).fit_transform(layout)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=layout_3d[:, 0],
                y=layout_3d[:, 1],
                z=layout_3d[:, 2],
                mode="markers",
                marker=dict(
                    size=12,
                    color=f,
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=True,
                ),
            )
        ]
    )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=700,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig.write_html(f"html_files/clusters/{file_name}_{row['cluster_id']}_{layer}.html")


def main():
    fix_umap_bug()
    layers = [
        # "inception3a",
        "inception3b",
        "inception4a",
        "inception4b",
        "inception4c",
        "inception4d",
        "inception4e",
        "inception5a",
        "inception5b",
    ]
    make_save_locations()
    for layer in layers:
        print(layer)
        plot_persistance_clusters(layer=layer, num_clusters=10, file_name="Test_html")


def make_save_locations(save_locations=["html_files/clusters/", "pdf_files/"]):
    for save_location in save_locations:
        if not os.path.exists(save_location):
            os.makedirs(save_location)


if __name__ == "__main__":
    main()
