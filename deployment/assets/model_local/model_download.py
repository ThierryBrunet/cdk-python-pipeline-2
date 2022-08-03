import json
import os


model_artifact = {"modelname": "lsmcrosswalk", "modelweights": [1, 2, 3, 4, 5]}
MODEL_OUTPUT_DIR = "/modeldata"
# MODEL_OUTPUT_DIR = "./"
if os.environ.get("HARSHIT_PATH") is not None:
    MODEL_OUTPUT_DIR = "deployment/assets"


print(model_artifact)

with open(os.path.join(MODEL_OUTPUT_DIR, "model.json"), "w") as f:
    json.dump(model_artifact, f, indent=2)
    print("new file has been created")

# docker build -t lsmtest:latest .
# docker run -v /Users/nyatih/github/lsm/deployment/assets/model_local:/modeldata lsmtest:latest
# docker run -v ./:/modeldata lsmtest:latest
