import aws_cdk as cdk
from .cross_account_resources_stack import CAResourcesStack
from constructs import Construct


class CAResourceStage(cdk.Stage):
    """
    This class contains the stage definition of cdk pipeline (resource).
    """

    def __init__(
        self, scope: Construct, id: str, bname: str, env: cdk.StackProps, **kwargs
    ):
        """
        Inits Resource class with scope, id, and props.

        :params scope: cdk app construct
        :params id: stack id/name
        :params props: additional properties passed to the stack
        """
        super().__init__(scope=scope, id=id, env=env, **kwargs)

        CAResourcesStack(
            self,
            "psdsdemocabucket2022",
            bname,
            env=cdk.Environment(account="881455463728", region="us-east-1"),
        )

