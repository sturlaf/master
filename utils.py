import numpy as np
from matplotlib import pyplot as plt


def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None,
    torus_colors=[],
    lw=2.5,
    cs=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
):

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = [
            "$H_0$",
            "$H_1$",
            "$H_2$",
            "$H_3$",
            "$H_4$",
            "$H_5$",
            "$H_6$",
            "$H_7$",
            "$H_8$",
        ]

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if len(plot_only) > 0:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    aspect = "equal"
    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
    #        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    i = 0
    for dgm, label in zip(diagrams, labels):
        c = cs[plot_only[i]]
        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none", c=c)
        i += 1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if len(torus_colors) > 0:
        # births1 = diagrams[1][:, 0]  # the time of birth for the 1-dim classes
        deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
        deaths1[np.isinf(deaths1)] = 0
        # lives1 = deaths1-births1
        # inds1 = np.argsort(lives1)
        inds1 = np.argsort(deaths1)
        ax.scatter(
            diagrams[1][inds1[-1], 0],
            diagrams[1][inds1[-1], 1],
            10 * size,
            linewidth=lw,
            edgecolor=torus_colors[0],
            facecolor="none",
        )
        ax.scatter(
            diagrams[1][inds1[-2], 0],
            diagrams[1][inds1[-2], 1],
            10 * size,
            linewidth=lw,
            edgecolor=torus_colors[1],
            facecolor="none",
        )

        # births2 = diagrams[2][
        #    :,
        # ]  # the time of birth for the 1-dim classes
        deaths2 = diagrams[2][:, 1]  # the time of death for the 1-dim classes
        deaths2[np.isinf(deaths2)] = 0
        # lives2 = deaths2-births2
        # inds2 = np.argsort(lives2)
        inds2 = np.argsort(deaths2)
        #        print(lives2, births2[inds2[-1]],deaths2[inds2[-1]], diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1])
        ax.scatter(
            diagrams[2][inds2[-1], 0],
            diagrams[2][inds2[-1], 1],
            10 * size,
            linewidth=lw,
            edgecolor=torus_colors[2],
            facecolor="none",
        )

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect(aspect, "box")

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="upper right")

    if show is True:
        plt.show()
    return ax
