import aws_cdk as cdk
from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    aws_s3 as s3,
    aws_codepipeline_actions as cp_actions,
    aws_codepipeline as cp,
    aws_codebuild as cb,
    aws_iam as iam,
)
from constructs import Construct


class AppPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        source_output = cp.Artifact()

        s3_bucket_info = s3.Bucket(
            self, id=f"{id}-s3", bucket_name="psdsprac2022-artificate"
        )

        cb_docker_build = cb.PipelineProject(
            self,
            "BuildlsmDockerImage",
            project_name="dslsmdemo",
            build_spec=cb.BuildSpec.from_source_filename(filename="buildspec.yml"),
            environment=cb.BuildEnvironment(privileged=True),
            description="Code build project for building lsm docker image",
            timeout=cdk.Duration.minutes(18),
        )

        # codebuild iam permissions to read write s3
        s3_bucket_info.grant_read_write(cb_docker_build)

        ## <<<<< PIPELINE ACTIONS >>>>> ##

        # Github Action
        github_source_action = cp_actions.GitHubSourceAction(
            action_name="GithubCodeSource",
            owner="harshitnyati",
            repo="lsm",
            oauth_token=cdk.SecretValue.secrets_manager(
                secret_id="dev/dssecrets", json_field="github-token"
            ),
            output=source_output,
            branch="main",
            trigger=cp_actions.GitHubTrigger.NONE,
        )

        # Build Action
        docker_build_action = cp_actions.CodeBuildAction(
            action_name="BuildDockerImage",
            input=source_output,
            project=cb_docker_build,
            run_order=1,
            type=cp_actions.CodeBuildActionType.BUILD,
        )

        # Source Artifact Stage
        source_stage = cp.StageProps(
            stage_name="Source", actions=[github_source_action]
        )

        # Build Stage
        build_stage = cp.StageProps(stage_name="Build", actions=[docker_build_action])

        ## <<<<<< CODE PIPELINE >>>>>> ##

        # define the pipeline
        pipeline = cp.Pipeline(
            self,
            "CodePipeline",
            cross_account_keys=False,
            pipeline_name=f"lsm-runtime-pipeline",
            artifact_bucket=s3_bucket_info,
            stages=[source_stage, build_stage],
        )

