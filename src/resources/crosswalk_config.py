crosswalk_config = {
    "local": {
        "s3_input_file_1": "../data/ab_GA_PM_SCI_df.csv",
        "s3_input_file_2": "../data/GA_SCI.csv",
        "s3_temp_processed_data": "../data/embeddings/",
        "s3_output_path": ".",
    },
    "dev": {
        "s3_input_file_1": "s3://{0}/{1}/file_1.csv",
        "s3_input_file_2": "s3://{0}/{1}/file_2.csv",
        "s3_temp_processed_data": "s3://{0}/{1}/",
        "s3_output_path": "",
    },
}
