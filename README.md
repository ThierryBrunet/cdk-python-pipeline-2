# lsm
first build

Machine Learning / AI papers with github code :- https://paperswithcode.com/ 

CDK bootstrap reference:- https://aws.plainenglish.io/cdk-cross-account-pipelines-3126e0434b0c

cdk bootstrap aws://890898987986/us-east-1 --profile jkhlhkh

cdk bootstrap aws://8769775648988/us-east-1 --profile jhkljg --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess' --trust 890898987986 --trust-for-lookups 890898987986


cdk bootstrap aws://765434679099/us-east-1 --profile jhkuull --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess' --trust 890898987986 --trust-for-lookups 890898987986

if any account/region is already bootstraped with older version for example v1 we just need to bootstrap that account/region again with latest cdk cli using command

cdk bootstrap aws://765434679099/us-east-1

AWS CDK Pipelines: Real-World Tips and Tricks (Part 2): https://levelup.gitconnected.com/aws-cdk-pipelines-real-world-tips-and-tricks-part-2-7a0d093a89a0

What is the difference between @aws-cdk/pipelines and @aws-cdk/aws-codepipeline?: https://stackoverflow.com/questions/69692259/what-is-the-difference-between-aws-cdk-pipelines-and-aws-cdk-aws-codepipeline

Workshops
CDK PIPELINES:- https://cdkworkshop.com/30-python/70-advanced-topics/200-pipelines.html
Why CDK Pipelines: https://www.antstack.io/blog/cdk-pipelines-with-github-source-and-codestar-connection/ 
CDK Pipelines: Continuous delivery for AWS CDK applications: https://aws.amazon.com/blogs/developer/cdk-pipelines-continuous-delivery-for-aws-cdk-applications/
Continuous integration and delivery (CI/CD) using CDK Pipelines: https://docs.aws.amazon.com/cdk/v2/guide/cdk_pipeline.html
CI/CD With AWS CodePipeline (CDK with manual approval python example): https://levelup.gitconnected.com/ci-cd-with-aws-codepipeline-a452c5b88c60 

Useful CDK Python repo links:- 
aws-cdk: https://github.com/aws/aws-cdk
aws-cdk-examples: https://github.com/aws-samples/aws-cdk-examples/blob/master/python/codepipeline-docker-build/Pipeline.py
zzenonn/InfrastructureCdkSample: https://github.com/zzenonn/InfrastructureCdkSample/blob/master/app.py


AWS Code Pipeline:-
Build a Cross-Account Continuous Delivery Pipeline Using AWS CodePipeline | Amazon Web Services: https://www.youtube.com/watch?v=PFQQkoO9kEc

AWS CLI:- 
Environment variables to configure the AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html
