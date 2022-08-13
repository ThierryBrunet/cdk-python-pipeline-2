from ensurepip import version
import logging

import aws_cdk as cdk
from aws_cdk import Stack
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_stepfunctions as sfn
from constructs import Construct
import json


class LSMResourcesStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, proj_config: dict, **kwargs
    ) -> None:
        """
        Initialize LSM Resources Stack.
        :param scope: cdk app construct (generated in app.py)
        :param id: stack logical id/name
        :param proj_config: project configuration passed to this stack from infra.yml and context variables
        """
        super().__init__(scope, construct_id, **kwargs)

        # prefix information is generated from the infra.yml
        prefix = proj_config.get("prefix")

        # env is generated from `--context`
        env = proj_config.get("env")
        s3_prefix = proj_config.get("s3_prefix")
        eventbus_prefix = proj_config.get("eventbus_prefix")
        state_machine_prefix = proj_config.get("state_machine_prefix")

        # create bucket for lsm S3 Resource
        logging.info("create bucket for lsm S3 Resource")
        # s3.Bucket(
        #     self,
        #     id=f"{id}-s3",
        #     bucket_name=f"{prefix}-{env}-{s3_prefix}",
        #     removal_policy=cdk.RemovalPolicy.RETAIN,
        #     encryption=s3.BucketEncryption.S3_MANAGED,
        #     block_public_access=s3.BlockPublicAccess(
        #         block_public_acls=True,
        #         block_public_policy=True,
        #         ignore_public_acls=True,
        #         restrict_public_buckets=True,
        #     ),
        # )

        self.eventbus_info = events.EventBus(
            self,
            id=f"{id}-eventbus",
            event_bus_name=f"{prefix}-{env}-{eventbus_prefix}-bus",
        )

        self.eventbus_info.apply_removal_policy(policy=cdk.RemovalPolicy.RETAIN)

        # events.CfnEventBusPolicy(
        #     self,
        #     id=f"{id}-eventbuspolicy",
        #     statement_id="AlloLSMAccountToPutEvents",
        #     action="events:PutEvents",
        #     event_bus_name=self.eventbus_info.event_bus_name,
        #     principal="504529306086",
        # )

        events.CfnEventBusPolicy(
            self,
            id=f"{id}-eventbuspolicy",
            statement_id="AlloLSMAccountToPutEventstest",
            action="events:PutEvents",
            event_bus_name=self.eventbus_info.event_bus_name,
            principal="*",
            # condition=events.CfnEventBusPolicy.ConditionProperty(
            #     key="aws:SourceArn", type="ArnEquals", value="o-1234567890",
            # ),
        )

        # events.CfnEventBusPolicy(
        #     self,
        #     id=f"{id}-eventbuspolicy",
        #     statement_id="AlloLSMAccountToPutEvents",
        #     action="events:PutEvents",
        #     event_bus_name=self.eventbus_info.event_bus_name,
        #     principal="*",
        #     condition=events.CfnEventBusPolicy.ConditionProperty(
        #         key="aws:PrincipalOrgID", type="StringEquals", value="o-1234567890",
        #     ),
        # )

        # events.CfnEventBusPolicy(
        #     self,
        #     id=f"{id}-eventbuspolicy",
        #     statement_id="AlloLSMAccountToPutEvents",
        #     action="events:PutEvents",
        #     event_bus_name=self.eventbus_info.event_bus_name,
        #     principal="*",
        #     statement={"Policy"},
        # )

        # events.CfnEventBusPolicy( it did not work
        #     self,
        #     id=f"{id}-eventbuspolicy",
        #     statement_id="AlloLSMAccountToPutEvents",
        #     action="events:PutEvents",
        #     event_bus_name=self.eventbus_info.event_bus_name,
        #     principal="504529306086",
        #     statement={"Principal": {"AWS": "arn:aws:iam::504529306086:*"}},
        # )

        # cdk.CfnResource(
        #     self,
        #     id=f"{id}-eventbuspolicy",
        #     type="AWS::Events::EventBusPolicy",
        #     properties={
        #         "StatementId": "AlloLSMAccountToPutEvents",
        #         "Statement": {
        #             "Effect": "Allow",
        #             "Principal": {
        #                 "AWS": [
        #                     "arn:aws:iam::504529306086:root",
        #                     "arn:aws:iam::504529306086:role/service-role/Service_RoleName",
        #                 ]
        #             },
        #             "Action": "events:PutEvents",
        #             "Resource": [eventbus_info.event_bus_arn],
        #         },
        #     },
        # )

        # cdk.CfnResource(
        #     self,
        #     id=f"{id}-eventbuspolicy",
        #     # version="2012-10-17",
        #     type="AWS::Events::EventBusPolicy",
        #     properties={
        #         "StatementId": "AlloLSMAccountToPutEventssss",
        #         "Statement": {
        #             "Effect": "Allow",
        #             "Principal": {"AWS": "arn:aws:iam::504529306086:root"},
        #             "Action": "events:PutEvents",
        #             "Resource": [self.eventbus_info.event_bus_arn],
        #         },
        #     },
        # )

        trigger_lsm_workflow = events.Rule(
            self,
            id=f"{id}-eventbusrule",
            event_bus=self.eventbus_info,
            event_pattern=events.EventPattern(
                source=["LSM"],
                detail_type=["IMPORT", "UPDATE"],
                detail=json.loads('{"districtUid": [{"exists": true}]}'),
                # detail={"districtUid": [{"exists": true}]},
            ),
            targets=[
                targets.SfnStateMachine(
                    machine=sfn.StateMachine.from_state_machine_arn(
                        self,
                        id=f"{id}-statemachine",
                        state_machine_arn=f"arn:aws:states:{self.region}:{self.account}:stateMachine:{prefix}-{env}-{state_machine_prefix}",
                    )
                )
            ],
        )

