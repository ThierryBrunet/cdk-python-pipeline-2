# lsm
first build

CDK bootstrap reference:- https://aws.plainenglish.io/cdk-cross-account-pipelines-3126e0434b0c

cdk bootstrap aws://890898987986/us-east-1 --profile jkhlhkh

cdk bootstrap aws://8769775648988/us-east-1 --profile jhkljg --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess' --trust 890898987986 --trust-for-lookups 890898987986


cdk bootstrap aws://765434679099/us-east-1 --profile jhkuull --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess' --trust 890898987986 --trust-for-lookups 890898987986