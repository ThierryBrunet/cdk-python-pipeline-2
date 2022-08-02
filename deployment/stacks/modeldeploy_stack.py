import aws_cdk as cdk
from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
)
from constructs import Construct
from os import path


class ModelDeployStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        MODEL_V1_KEY_PREFIX = "model-v1"

        s3_deploy.BucketDeployment(
            self,
            "model_deploy",
            sources=[
                s3_deploy.Source.asset(
                    path=path.join("assets", "model_v1"),
                    bundling=cdk.BundlingOptions(
                        image=cdk.DockerImage.from_build(
                            path=path.abspath(path.join("assets", "model_v1"))
                        ),
                        user="root",
                        command=["python", "build.py"],
                    ),
                )
            ],
            destination_key_prefix=MODEL_V1_KEY_PREFIX,
            destination_bucket="s3://psdsprac2022/",
            memory_limit=2048,
        )
