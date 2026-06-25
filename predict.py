import os
import subprocess

# MODEL = "M2_NS"
for model in ["NS_split"]:
    for seed in [1, 2, 3]:
        model_path = f"checkpoints/{model}_{seed}.ckpt"
        if os.path.exists(model_path):
            config = {
                "predict_parameters_path": f"training_config/{model}.json",
                "model_path": model_path,
                "prediction_save_path": f"predictions/{model}_{seed}",
                "no_edge_feature": None,
            }

            # Build the command
            cmd = ["python", "-m", "graphphysics.predict"]
            for key, value in config.items():
                if value is not None:
                    cmd.append(f"--{key}={value}")
                else:
                    cmd.append(f"--{key}")

            # Run it
            result = subprocess.run(cmd, check=True)

print("done")
