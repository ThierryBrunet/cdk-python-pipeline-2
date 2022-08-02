import aws_cdk as cdk
from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    aws_s3 as s3,
)
from constructs import Construct


class CAResourcesStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, bname: str, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        # example resource
        # queue = sqs.Queue(
        #     self, "AppQueue",
        #     visibility_timeout=Duration.seconds(300),
        # )
        s3.Bucket(
            self,
            f"{bname}2022",
            bucket_name=bname,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
