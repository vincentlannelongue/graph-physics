import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from phd_utils import (
    merge_prediction_files,
    compute_rmse,
    # compute_normalised_rmse,
    # compute_WSS,
    # compute_shear_rate,
    xdmf_to_meshes,
    # carreau_yasuda_law,
    # meshes_to_xdmf,
    add_wss_on_merged_file,
    compute_rmse_gp
)

FIELDS_MAP = {"Vitesse": ["x0", "x1", "x2"]}
# FIELDS_MAP = {"Vitesse": ["x0", "x1", "x2"], "Pression": ["x3"]}
TESTSET_FINE_PATH = "/scratch-big/vlannelongue1/00_Data/Datasets/NoStent/test/"
TESTSET_COARSE_PATH = "/scratch-big/vlannelongue1/00_Data/Datasets/NoStentCoarse/test/"

DF_PATH = "test_results.csv"
PREDICTION_PATH = "predictions"
ADDITIONAL_TRUTH_FIELDS = []

ANEURYSM_NT = 7


# TODO: add aneurysm metrics

if os.path.exists(DF_PATH):
    df = pd.read_csv(DF_PATH)
else:
    df = pd.DataFrame(
        columns=[
            "Model",
            "RMSE_Vitesse",
            "RMSE_GP",
            "RMSE_Vitesse_std",
            "RMSE_Vitesse_Aneurysm",
            "RMSE_Vitesse_Aneurysm_std",
            "RMSE_Pression",
            "RMSE_Pression_std",
            "RMSE_WSS",
            "RMSE_WSS_std",
        ]
    )

for model in ["NS_norm_1"]:  # os.listdir(PREDICTION_PATH):  # "NS_baseline_2",
    if "NSC" in model:
        testset_path = TESTSET_COARSE_PATH
    else:
        testset_path = TESTSET_FINE_PATH

    if model in df["Model"].values:
        print(f"Results for {model} already exist in {DF_PATH}, skipping...")
        continue

    print(f"Processing model: {model}")

    prediction_save_path = f"predictions/{model}"
    out_dir = f"results/{model}/predictions_merged"
    os.makedirs(out_dir, exist_ok=True)

    rmse_v_rollout = []
    rmse_gp = []
    rmse_v_rollout_aneurysm = []
    rmse_p_rollout = []
    rmse_wss_rollout = []

    for file in tqdm(os.listdir(testset_path)):
        if file.endswith(".xdmf"):
            case_name = os.path.splitext(file)[0]
            truth_path = os.path.join(testset_path, file)
            out_file = os.path.join(out_dir, case_name)

            # print(xdmf_to_meshes(truth_path)[0].point_data.keys())
            # print(xdmf_to_meshes(truth_path)[0].point_data.keys())
            nodetype = xdmf_to_meshes(truth_path)[0].point_data["node_type"]
            aneurysm_mask = nodetype == ANEURYSM_NT
            if os.path.exists(f"{out_file}.xdmf"):
                merged_meshes = xdmf_to_meshes(f"{out_file}.xdmf")
            else:
                pred_path = os.path.join(
                    prediction_save_path, f"graph_{case_name}.xdmf"
                )
                merged_meshes = merge_prediction_files(
                    truth_path,
                    pred_path,
                    fields_map=FIELDS_MAP,
                    verbose=True,
                    out_path=out_file,
                    delay=0,
                    additional_truth_fields=ADDITIONAL_TRUTH_FIELDS,
                )
            # if ("WSS" not in FIELDS_MAP.keys()) and (
            #     "WSS_Prediction" not in merged_meshes[0].point_data.keys()
            # ):
            #     merged_meshes = add_wss_on_merged_file(
            #         file_path=os.path.join(out_dir, f"{case_name}.xdmf"),
            #         velocity_field_name="Vitesse",
            #         wss_field_name="WSS",
            #         out_path=out_file,
            #         verbose=True,
            #     )

            rmse_gp.append(compute_rmse_gp(meshes=merged_meshes))

            rmse_v_rollout.append(compute_rmse(merged_meshes, fieldname="Vitesse"))
            rmse_v_rollout_aneurysm.append(
                compute_rmse(merged_meshes, fieldname="Vitesse", mask=aneurysm_mask)
            )

            if "Pression" in FIELDS_MAP.keys():
                rmse_p_rollout.append(compute_rmse(merged_meshes, fieldname="Pression"))
            # rmse_wss_rollout.append(compute_rmse(merged_meshes, fieldname="WSS"))

    rmse_v_rollout_mean = np.mean(rmse_v_rollout)
    rmse_v_rollout_std = np.std(rmse_v_rollout)

    rmse_gp_mean = np.mean(rmse_gp)

    rmse_v_rollout_aneurysm_mean = np.mean(rmse_v_rollout_aneurysm)
    rmse_v_rollout_aneurysm_std = np.std(rmse_v_rollout_aneurysm)

    rmse_p_rollout_mean = np.mean(rmse_p_rollout)
    rmse_p_rollout_std = np.std(rmse_p_rollout)

    rmse_wss_rollout_mean = np.mean(rmse_wss_rollout)
    rmse_wss_rollout_std = np.std(rmse_wss_rollout)

    new_row = pd.DataFrame(
        [
            {
                "Model": model,
                "RMSE_Vitesse": f"{rmse_v_rollout_mean:.4f}",
                "RMSE_GP": f"{rmse_gp_mean:.4f}",
                "RMSE_Vitesse_std": f"{rmse_v_rollout_std:.4f}",
                "RMSE_Vitesse_Aneurysm": f"{rmse_v_rollout_aneurysm_mean:.4f}",
                "RMSE_Vitesse_Aneurysm_std": f"{rmse_v_rollout_aneurysm_std:.4f}",
                "RMSE_Pression": (
                    f"{rmse_p_rollout_mean:.4f}" if rmse_p_rollout else None
                ),
                "RMSE_Pression_std": (
                    f"{rmse_p_rollout_std:.4f}" if rmse_p_rollout else None
                ),
                "RMSE_WSS": f"{rmse_wss_rollout_mean:.4f}",
                "RMSE_WSS_std": f"{rmse_wss_rollout_std:.4f}",
            }
        ]
    )

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DF_PATH, index=False)


print("done")
