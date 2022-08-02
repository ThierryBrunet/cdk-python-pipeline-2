from os import path
import tarfile
import os
import zipfile

print(path.join("deployment", "assets", "model_v1"))
print(tarfile)

OUTPUT_DIR = "/Users/nyatih/github/lsm/deployment/assets/"
STAGING_DIR = "/Users/nyatih/github/lsm/deployment/assets/model_v1/code/"


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path)),
            )


with zipfile.ZipFile(
    "/Users/nyatih/github/lsm/deployment/assets/model.zip", "w", zipfile.ZIP_DEFLATED
) as zipf:
    zipdir("/Users/nyatih/github/lsm/deployment/assets/model_v1/code/", zipf)

# with tarfile.open(os.path.join(OUTPUT_DIR, "model.tar.gz"), mode="w:gz") as archive:
#     archive.add(STAGING_DIR, recursive=True)

