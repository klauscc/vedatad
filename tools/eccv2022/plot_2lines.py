#!/usr/bin/env python

import matplotlib.pyplot as plt


def plot_2lines(
    x: list,
    y1: list,
    y2: list,
    x_label: str,
    y1_label: str,
    y2_label: str,
    save_path,
    plot_type="plot",
    ax1_annots=None,
    ax2_annots=None,
):
    """plot figure with two lines in different scale.

    Args:
        x (list): the x-axis data.
        y1 (list): the y1-axis data.
        y2 (list): the y2-axis data.
        x_label (str): x-axis label.
        y1_label (str): y1-axis label.
        y2_label (str): y2-axis label.

    """
    fig, ax = plt.subplots(figsize=(3.5,3.5), dpi=400)

    getattr(ax, plot_type)(x, y1, color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_ylabel(y1_label)
    ax.set_xlabel(x_label)
    gap = (max(y1) - min(y1)) * 0.2
    ax.set_ylim(min(y1) - gap, max(y1) + gap)

    # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()
    getattr(ax2, plot_type)(x, y2, color="green")
    gap = (max(y2) - min(y2)) * 0.25
    ax2.set_ylim(min(y2) - gap, max(y2) + gap)
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylabel(y2_label)

    # if ax2_annots is not None:
    #     for i, j, text in ax2_annots:
    #         ax2.annotate(text, xy=(i, j))

    plt.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close()


# =======================================================


def temporal_extent_vs_mAP_memory():
    save_path = "eccv2022/figs/temporal_extent__vs__map_memory.png"
    x = ["8", "16", "24", "32", "40"]
    y1 = [41.5, 49.7, 51.5, 52.7, 53.3]
    y2 = [4, 8, 10, 15, 18]
    plot_2lines(
        x, y1, y2, "temporal extent (s)", "mAP(%)", "GPU Mem (GB/GPU)", save_path
    )


def fps_vs_mAP_memory():
    save_path = "eccv2022/figs/fps__vs__map_memory.png"
    x = ["8", "15"]
    y1 = [48.6, 52.7]
    y2 = [8, 15]
    plot_2lines(x, y1, y2, "fps", "mAP(%)", "GPU Mem (GB/GPU)", save_path, "scatter")


def backbone_vs_mAP_memory():
    save_path = "eccv2022/figs/backbone__vs__map_memory.png"
    x = ["I3D-112", "Swin-T-112", "Swin-B-112", "Swin-B-224"]
    y1 = [49.9, 53.2, 55.3, 59]
    y2 = [12, 26, 35, 48]
    plot_2lines(
        x,
        y1,
        y2,
        "backbone",
        "mAP(%)",
        "GPU Mem (GB/GPU)",
        save_path,
        plot_type="bar",
    )


def resolution_vs_mAP_memory():
    save_path = "eccv2022/figs/resolution__vs__map_memory.png"
    x = ["112x112", "168x168", "224x224", "336x336"]
    y = [19, 42, 76, 168]
    fig = plt.figure(figsize=(3.5,3.5))
    plt.plot(x, y, marker="o")
    plt.xlabel("spatial resolution")
    plt.ylabel("GPU Mem (GB/GPU)")
    plt.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close()


def frozen_stages_vs_mAP_memory():
    save_path = "eccv2022/figs/frozen_stages__vs__map_memory.png"
    x = ["0", "2", "3", "4"]
    y1 = [53.2, 52.8, 48.4, 44.3]
    y2 = [26, 15, 8, 3]
    plot_2lines(
        x,
        y1,
        y2,
        "backbone",
        "mAP(%)",
        "GPU Mem (GB/GPU)",
        save_path,
        plot_type="plot",
    )


if __name__ == "__main__":
    temporal_extent_vs_mAP_memory()
    fps_vs_mAP_memory()
    backbone_vs_mAP_memory()
    resolution_vs_mAP_memory()
    frozen_stages_vs_mAP_memory()
