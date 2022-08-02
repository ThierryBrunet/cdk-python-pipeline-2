# download sentence transformer model and create model.tar.gz package file
import os
import tarfile
from sentence_transformers import models, SentenceTransformer


STAGING_DIR = "/transformer"  # model files are downloaded here
CODE_DIR = "code"  # code files (e.g. inference.py, requirements.txt) are available here
OUTPUT_DIR = "/asset-output"  # model archive file is placed here

if not os.path.isdir(STAGING_DIR):
    os.makedirs(STAGING_DIR)

model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
model.save(STAGING_DIR)

with tarfile.open(os.path.join(OUTPUT_DIR, "model.tar.gz"), mode="w:gz") as archive:
    archive.add(STAGING_DIR, recursive=True)
    # archive.add(CODE_DIR, recursive=True)
