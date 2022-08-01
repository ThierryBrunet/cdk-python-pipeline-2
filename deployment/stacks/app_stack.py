import aws_cdk as cdk
from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    # aws_s3 as s3,
    pipelines,
)
from constructs import Construct
from .resources_stage import ResourceStage


class AppStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        cp_source = pipelines.CodePipelineSource.git_hub(
            repo_string="harshitnyati/lsm",
            branch="main",
            authentication=cdk.SecretValue.secrets_manager(
                secret_id="dev/dssecrets", json_field="github-token"
            ),
        )

        cdk_pipeline = pipelines.CodePipeline(
            self,
            id=construct_id,
            pipeline_name="dsdemopipeline",
            synth=pipelines.ShellStep(
                "Synth",
                input=cp_source,
                commands=[
                    "npm install -g aws-cdk",
                    "ls",
                    "cd deployment",
                    "pip install -r requirements.txt",
                    "cdk synth",
                ],
                primary_output_directory="deployment/cdk.out",
            ),
        )

        resource_stage = ResourceStage(self, "psdsdemoresourcestage2022")
        cdk_pipeline.add_stage(resource_stage)
