"""
Matching script
"""

import os
import utils

#### inputs via environment variables ####

# bucket for reading/writing intermediate data
data_bucket = os.environ['data_bucket']
# unique execution id token
execution_id = os.environ.get('execution_id')
# s3 url where input standards file can be found
input_standards_s3_url = os.environ['input_standards_s3_url']
# input standards type (sis, case)
input_standards_type = os.environ['input_standards_type']
# s3 url for index to match to
index_s3_location = os.environ['index_s3_location']
# s3 url to write results to
automated_mapping_s3_url = os.environ['automated_mapping_s3_url']


print(f'Processing input: {input_standards_s3_url}')

# handle input type
if input_standards_type == 'SIS':
    match_input = utils.StandardSet.from_SIS_csv(
        input_standards_s3_url
    )
elif input_standards_type == 'CASE':
    match_input = utils.StandardSet.from_CASE_json(
        input_standards_s3_url
    )
else:
    raise NotImplementedError('input type not recognized')

# index standards to match to
match_index = utils.AbMatchIndex(
    index_s3_location
)

# run matching
matcher = utils.AutomatedMatchCalculator(
    match_input=match_input,
    match_index=match_index,
    s3_data_bucket=data_bucket,
    embedding_job_params=dict(
        model_name=utils.get_parameter('/standards-match/model_name'),
        job_name = f'{execution_id}-description'
    )
)
# calculate similarity scores
similarity = matcher.calculate()

# find top matches and organize as tabular report
df_report = matcher.find_top_matches(similarity)

df_report.to_csv(automated_mapping_s3_url, index=False)
