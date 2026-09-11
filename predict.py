import json
import os
import subprocess

# MODEL = "M2_NS"
for model in ["KDC_baseline"]:
    for seed in [2, 3]:
        model_path = f"checkpoints/{model}_{seed}.ckpt"
        if os.path.exists(model_path):
            # Check inference path
            parameters_path = f"training_config/{model}.json"
            with open(parameters_path, "r") as fp:
                parameters = json.load(fp)
            if "test" in parameters["dataset"]["test_path"]:
                raise ValueError(
                    f"Test path in {parameters_path} is set to 'test', which is not allowed for inference."
                )
            config = {
                "predict_parameters_path": parameters_path,
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
