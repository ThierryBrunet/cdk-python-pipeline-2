"""
Build index - sm processing script
Builds AB index for a state from a full csv dataset of Academic Benchmark standards

Assumes SSM parameters are populated:
- /standards-match/model_name - name of SageMaker model to use for feature calculation
- /standards-match/data_bucket - name of bucket to store intermediate inference results in
"""

import os
import pandas as pd

# standards matching project utils
import utils


state_key = os.environ['state_key']
index_s3_location = os.environ['index_s3_location']  # constructed in prep lambda
ab_data_s3_url = os.environ['ab_data_s3_url']  # s3 url of full AB csv dataset, passed from sf input

# load in AB dataset
df_ab = pd.read_csv(ab_data_s3_url)

# extract subset by state
df_subset = df_ab.loc[lambda x: x.state_abbr==state_key]
standard_set = utils.StandardSet.from_AB_df(df_subset)

# populate index in specified s3 location

utils.AbMatchIndex.build(
    index_s3_location=index_s3_location, 
    standard_set = standard_set,
    embedding_job_params=dict(
        model_name=utils.get_parameter('/standards-match/model_name'),
        data_bucket=utils.get_parameter('/standards-match/data_bucket'),
        job_name_suffix=f'-index-{state_key}-description',
    ),
    overwrite=True
)

