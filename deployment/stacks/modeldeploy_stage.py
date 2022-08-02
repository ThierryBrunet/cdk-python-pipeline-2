import aws_cdk as cdk
from .modeldeploy_stack import ModelDeployStack
from constructs import Construct


class ModelDeployStage(cdk.Stage):
    """
    This class contains the stage definition of cdk pipeline (resource).
    """

    def __init__(self, scope: Construct, id: str, **kwargs):
        """
        Inits Resource class with scope, id, and props.

        :params scope: cdk app construct
        :params id: stack id/name
        :params props: additional properties passed to the stack
        """
        super().__init__(scope=scope, id=id, **kwargs)

        ModelDeployStack(self, "psdslsmmodel2022")

