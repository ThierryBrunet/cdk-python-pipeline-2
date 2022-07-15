# -*- coding: utf-8 -*-


import os

import pandas as pd

from preprocessing import Preprocessing
from resources.config import Config

os.chdir("./src")
environment = os.environ.get("ENV", "").lower()
if environment == "":
    environment = "local"
conf = Config(environment)
config = conf.get_config()


## read dataframes:
df1 = pd.read_csv(config[scope]["s3_input_file_1"])
df2 = pd.read_csv(config["s3_input_file_2"], encoding="unicode_escape")


## prep for preprocessing calls
file_name_1 = config["s3_input_file_1"].split("/")[-1].replace(".csv", "")
file_name_2 = config["s3_input_file_2"].split("/")[-1].replace(".csv", "")


## NEW GRADES custom, create column in preprocessing
column_dict_1 = {
    "id": "GUID",
    "parent_id": "Parent_GUID",
    "subject": "Subject",
    "grade_level": "NEW_GRADES",
    "standard": "Description",
}
column_dict_2 = {
    "id": "identifier",
    "parent_id": "",
    "subject": "",
    "grade_level": "educationLevel",
    "standard": "fullStatement",
}


## Preprocessing calls
# File 1
i1_preprocessing = Preprocessing(df1, "AB", file_name_1, column_dict_1, config)
print(" AB file preprocessing successfully done! ")
# File 2
i2_preprocessing = Preprocessing(df2, "CASE", file_name_2, column_dict_2, config)
print(" CASE file preprocessing successfully done! ")


# i2_preprocessing.ls_tree.jsn_data
# i2_preprocessing.embeddings_path


# import os
# print(os.environ)
