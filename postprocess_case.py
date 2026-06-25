import numpy as np
import os

import matplotlib.pyplot as plt
from typing import Tuple
from phd_utils import extract_point_values, xdmf_to_meshes, plot_point_comparison

MODEL = "NS_split_3"
FIELDS_MAP = {"Vitesse": ["x0", "x1", "x2"]}
# FIELDS_MAP = {"Vitesse": ["x0", "x1", "x2"], "Pression": ["x3"]}
CASE_NAME = "saura_209"
POINTS_VEL = {
    "aneurysm_center": [5.573, 9.644, 6.348],
    "aneurysm_close_to_wall": [5.848, 10.647, 6.348],
    "neck_center": [6.003, 7.897, 8.575],
    "neck_close_to_wall": [6.947, 8.687, 6.354],
    "outlet_center": [3.821, 9.806, -1.302],
    "outlet_close_to_wall": [5.236, 10.869, -1.740],
}

POINTS_WSS = {
    "aneurysm_wall": [5.855, 10.661, 6.328],
    "neck_wall": [6.949, 8.709, 6.343],
    "outlet_wall": [5.256, 10.883, -1.747],
}

OUT_PATH = f"results/{MODEL}/plot"
PREDICTION_PATH = f"results/{MODEL}/predictions_merged/{CASE_NAME}.xdmf"

# for model in os.listdir(PREDICTION_PATH):
meshes = xdmf_to_meshes(PREDICTION_PATH)

# fig_wss, axes_wss = plt.subplots(ncols=len(POINTS_WSS.keys()), nrows=1)
# for i, point in enumerate(POINTS_WSS.keys()):
#     point_data = extract_point_values(meshes=meshes, coords=POINTS_WSS[point])
#     wss_pred = np.linalg.norm(np.array(point_data["WSS_Prediction"]), axis=1)
#     wss_truth = np.linalg.norm(np.array(point_data["WSS_Truth"]), axis=1)
#     axes_wss[i].plot(wss_pred, label="pred")
#     axes_wss[i].plot(wss_truth, label="truth", linestyle="dashed")
#     axes_wss[i].set_title(point)
#     axes_wss[i].legend()
# plt.tight_layout()
# plt.savefig(f"{OUT_PATH}_wss.png")


def plot_point_comparison(
    meshes,
    points: dict,
    field_name: str,
    out_path: str = "plot.png",
    nrows: int = 1,
    figsize: Tuple = (15, 10),
) -> None:
    """
    Plot the comparison of predicted and true values of a given field at specific points over time.

    Args:
        meshes: List of meshio meshes containing the point data to plot.
        points: Dict of point names and their coordinates. For instance, {"aneurysm_center": [5.573, 9.644, 6.348]}.
        field_name: Name of the field to plot (e.g., "Vitesse" or "WSS"). The function will look for f"{field_name}_Prediction" and f"{field_name}_Truth" in the point data.
        out_path: Path to save the resulting plot.
        nrows: Number of rows in the subplot grid.
        figsize: Size of the figure.
    """
    fig, axes = plt.subplots(
        ncols=len(points.keys()) // nrows, nrows=nrows, squeeze=False, figsize=figsize
    )

    for n, point in enumerate(points.keys()):
        i = n % nrows
        j = n // nrows
        point_data = extract_point_values(meshes=meshes, coords=points[point])
        pred = np.linalg.norm(np.array(point_data[f"{field_name}_Prediction"]), axis=1)
        truth = np.linalg.norm(np.array(point_data[f"{field_name}_Truth"]), axis=1)
        error = abs(pred - truth)
        axes[i, j].plot(pred, label="pred")
        axes[i, j].plot(truth, label="truth", linestyle="dashed")
        axes[i, j].plot(error, label="error", linestyle="dotted")
        axes[i, j].set_title(point)
        axes[i, j].legend()
    plt.tight_layout()
    plt.savefig(out_path)


plot_point_comparison(
    meshes=meshes,
    points=POINTS_VEL,
    field_name="Vitesse",
    out_path=f"{OUT_PATH}_vel.png",
    nrows=2,
    figsize=(15, 10),
)
plot_point_comparison(
    meshes=meshes,
    points=POINTS_WSS,
    field_name="WSS",
    out_path=f"{OUT_PATH}_wss.png",
    nrows=1,
    figsize=(15, 5),
)

print("done")
