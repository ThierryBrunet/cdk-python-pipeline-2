import aws_cdk as cdk
from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    # aws_s3 as s3,
    pipelines,
    aws_iam as iam,
    aws_codebuild as cb,
)
from constructs import Construct
from .resources_stage import ResourceStage
from .apppipeline_stage import AppPipelineStage
from .cross_account_resources_stage import CAResourceStage
from .modeldeploy_stage import ModelDeployStage


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
            # synth=pipelines.ShellStep(
            #     "Synth",
            #     input=cp_source,
            #     commands=[
            #         "npm install -g aws-cdk@2.33.0",
            #         "ls",
            #         "cd deployment",
            #         "pip install -r requirements.txt",
            #         "cdk synth",
            #     ],
            #     primary_output_directory="deployment/cdk.out",
            # ),
            synth=pipelines.CodeBuildStep(
                id=f"lsm-cdkpipeline-dev",
                input=cp_source,
                commands=[
                    "npm install -g aws-cdk",
                    "ls",
                    "cd deployment",
                    "pip install -r requirements.txt",
                    "cdk synth",
                ],
                build_environment=cb.BuildEnvironment(privileged=True),
                primary_output_directory="deployment/cdk.out",
                role_policy_statements=[
                    iam.PolicyStatement(
                        resources=["*"],
                        actions=[
                            "s3:*",
                            "cloudformation:DescribeStacks",
                            "cloudformation:DeleteChangeSet",
                            "iam:PassRole",
                            "cloudformation:CreateChangeSet",
                            "cloudformation:DescribeChangeSet",
                            "cloudformation:ExecuteChangeSet",
                            "sts:AssumeRole",
                        ],
                        effect=iam.Effect.ALLOW,
                    )
                ],
            ),
        )

        # manual approval step
        manual_approvalstage = pipelines.ManualApprovalStep("Approve")
        resource_stage = ResourceStage(self, "psdsdemoresourcestage2022")
        cdk_pipeline.add_stage(resource_stage, post=[manual_approvalstage])

        # manual Approval step
        new_manual_approvalstage = pipelines.ManualApprovalStep("Approve")
        caresource_stage = CAResourceStage(
            self,
            id="psdsdemocaresources2022",
            bname="prodpsdsprac2022",
            env=cdk.Environment(account="881455463728", region="us-east-1"),
        )
        cdk_pipeline.add_stage(caresource_stage, post=[new_manual_approvalstage])

        # deploy_model = ModelDeployStage(self, "psmodeldeploymentstage2022")

        # cdk_pipeline.add_stage(deploy_model)

        apppipeline_stage = AppPipelineStage(self, "psdsdemoapppipelinestage2022")
        cdk_pipeline.add_stage(apppipeline_stage)

