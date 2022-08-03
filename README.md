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

Multi Environment Deployments using AWS CDK: https://www.youtube.com/watch?v=H7Ynxkk_jss

Workshops
CDK PIPELINES:- https://cdkworkshop.com/30-python/70-advanced-topics/200-pipelines.html
Why CDK Pipelines: https://www.antstack.io/blog/cdk-pipelines-with-github-source-and-codestar-connection/ 
CDK Pipelines: Continuous delivery for AWS CDK applications: https://aws.amazon.com/blogs/developer/cdk-pipelines-continuous-delivery-for-aws-cdk-applications/
Continuous integration and delivery (CI/CD) using CDK Pipelines: https://docs.aws.amazon.com/cdk/v2/guide/cdk_pipeline.html
CI/CD With AWS CodePipeline (CDK with manual approval python example): https://levelup.gitconnected.com/ci-cd-with-aws-codepipeline-a452c5b88c60 
Building a Secure Cross-Account Continuous Delivery Pipeline (old): https://aws.amazon.com/blogs/devops/aws-building-a-secure-cross-account-continuous-delivery-pipeline/ 

Useful CDK Python repo links:- 
aws-cdk: https://github.com/aws/aws-cdk
aws-cdk-examples: https://github.com/aws-samples/aws-cdk-examples/blob/master/python/codepipeline-docker-build/Pipeline.py
zzenonn/InfrastructureCdkSample: https://github.com/zzenonn/InfrastructureCdkSample/blob/master/app.py
aws-cdk-mwaa: https://github.com/ramonmarrero/aws-cdk-mwaa
aws-lambda-container-cdk: https://github.com/ramonmarrero/aws-lambda-container-cdk
AWS CDK Cross-Account Pipeline Demo: https://github.com/markilott/aws-cdk-pipelines-demo

Useful CDK Python Blogs link:-
Deploying Amazon Managed Apache Airflow with AWS CDK: https://medium.com/geekculture/deploying-amazon-managed-apache-airflow-with-aws-cdk-7376205f0128
Deploying AWS Lambda layers with Python: https://medium.com/geekculture/deploying-aws-lambda-layers-with-python-8b15e24bdad2
Deploying AWS Lambda Container Images: https://medium.com/geekculture/deploying-aws-lambda-container-images-43a9e37b25ab
Deploying a Simple CI/CD Pipeline using AWS CDK (Python): https://towardsdev.com/deploying-a-simple-ci-cd-pipeline-using-aws-cdk-python-748a77e8a6c2

Useful CDK Typescript repo links:- 
aws-cdk-pipelines-github: https://github.com/markilott/aws-cdk-pipelines-github/blob/main/lib/application/application-stage.ts


AWS Code Pipeline:-
Build a Cross-Account Continuous Delivery Pipeline Using AWS CodePipeline | Amazon Web Services: https://www.youtube.com/watch?v=PFQQkoO9kEc
Reference Architecture: Cross Account AWS CodePipeline: https://github.com/awslabs/aws-refarch-cross-account-pipeline

AWS CLI:- 
Environment variables to configure the AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html

AWS MWAA:- 
https://github.com/aws/aws-mwaa-local-runner/blob/main/docker/config/requirements.txt
Installing Python dependencies: https://docs.aws.amazon.com/mwaa/latest/userguide/working-dags-dependencies.html#working-dags-dependencies-mwaaconsole-version

Sagemaker:- 
ChandraLingam/AmazonSageMakerCourse: https://github.com/ChandraLingam/AmazonSageMakerCourse/blob/master/CustomAlgorithm/TensorFlow/Iris/iris_tensorflow_training_and_serving.ipynb
sagemaker-python-sdk: https://github.com/aws/sagemaker-python-sdk
sagemaker-containers: https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
SageMaker Clarify: https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-clarify/index.html#sagemaker-clarify
Sagemaker Clarify: https://github.com/aws/amazon-sagemaker-clarify/tree/master/src/smclarify
SHAP Baselines for Explainability: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-feature-attribute-shap-baselines.html
Fairness and Explainability with SageMaker Clarify: https://sagemaker-examples.readthedocs.io/en/latest/sagemaker_processing/fairness_and_explainability/fairness_and_explainability.html
Automating Amazon SageMaker with Amazon EventBridge: https://docs.aws.amazon.com/sagemaker/latest/dg/automating-sagemaker-with-eventbridge.html
Log Amazon SageMaker API Calls with AWS CloudTrail: https://docs.aws.amazon.com/sagemaker/latest/dg/logging-using-cloudtrail.html

MLOps foundation roadmap for enterprises with Amazon SageMaker: https://aws.amazon.com/blogs/machine-learning/mlops-foundation-roadmap-for-enterprises-with-amazon-sagemaker/

Getting-Started-with-Amazon-SageMaker-Studio: https://github.com/PacktPublishing/Getting-Started-with-Amazon-SageMaker-Studio



Sagemaker Books :- 
Getting Started with Amazon SageMaker Studio: https://learning.oreilly.com/library/view/getting-started-with/9781801070157/
Chapter 11: Operationalize ML Projects with SageMaker Projects, Pipelines, and Model Registry: https://learning.oreilly.com/library/view/getting-started-with/9781801070157/B17447_11_ePub_RK.xhtml#_idParaDest-146
sagemaker pipeline ModelQualityCheckConfig(: https://books.google.co.in/books?id=dJ1hEAAAQBAJ&pg=PA278&lpg=PA278&dq=sagemaker+pipeline+ModelQualityCheckConfig(&source=bl&ots=nPnXTbgayI&sig=ACfU3U3mSjTcwxR909vxo4UkE6iHlG9wFA&hl=en&sa=X&ved=2ahUKEwjezJXgqYf4AhVC7HMBHSehBrcQ6AF6BAgCEAM#v=onepage&q&f=false


SAM:- 
bharaththippireddy/serverlessusingawslambdaforpythondevelopers: https://github.com/bharaththippireddy/serverlessusingawslambdaforpythondevelopers

DynamoDB:- 
Advanced Design Patterns for DynamoDB - AWS Virtual Workshop: https://www.youtube.com/watch?v=dpfoHyvI8Dc

ELastic Search:- 
Elastcisearch Tutorial | Elk Stack: https://www.youtube.com/watch?v=WDIrz1nblk0&list=PLGZAAioH7ZlO7AstL9PZrqalK0fZutEXF&index=5
How to start logstash and converting log data into structure format | Logstash tutorial: https://www.youtube.com/watch?v=zO83-5-pcqw&list=PL3GCZkoyKK4dJd21JQT234gk62FxSlRhx&index=3
How to Use Logstash to import CSV Files Into ElasticSearch: https://www.youtube.com/watch?v=_kqunm8w7GI&list=PLS1QulWo1RIYkDHcPXUtH4sqvQQMH3_TN&index=2
Amazon Elasticsearch Service Deep Dive - AWS Online Tech Talks: https://www.youtube.com/watch?v=SOTFnRezIH0
Logstash Elasticsearch Kibana Tutorial | Logstash pipeline & input, output configurations.: https://www.youtube.com/watch?v=G2TUmPZ1slw
Introduction into the Elasticsearch Python Client: https://www.youtube.com/watch?v=UWR9G-U88X0
ElasticSearch with Python: https://www.youtube.com/watch?v=ma3BC8aPBfE
AWS Tutorials - Absolute Beginners Tutorial for Amazon Elasticsearch: https://www.youtube.com/watch?v=dPCFGt1AuJw


ECS:- 
AWS Step Function Integration with ECS or Fargate Tasks: Data In and Out: https://nuvalence.io/blog/aws-step-function-integration-with-ecs-or-fargate-tasks-data-in-and-out
How to use AWS Fargate and Lambda for long-running processes in a Serverless app: https://www.serverless.com/blog/serverless-application-for-long-running-process-fargate-lambda/
stacksimplify/aws-fargate-ecs-masterclass: https://github.com/stacksimplify/aws-fargate-ecs-masterclass
AWS ECS Networking Deep Dive & Service Discovery: https://www.youtube.com/watch?v=2fwCg82pMI4 
Udemy Courses:- AWS ECS (Elastic Container Service) - Deep Dive , AWS Fargate & ECS - Masterclass | Microservices, Docker, CFN



API:-
API Composition Pattern with GraphQL: https://www.linkedin.com/pulse/api-composition-pattern-graphql-domenico-vacchiano/

Deep Learning:- 
Practical Deep Learning: https://www.youtube.com/watch?v=8SF_h3xF3cE&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU
What is backpropagation really doing?: https://www.youtube.com/watch?v=Ilg3gGewQ5U
MIT 6.S191 (2019): Introduction to Deep Learning: https://www.youtube.com/watch?v=5v1JnYv_yWs
GANs for Good- A Virtual Expert Panel by DeepLearning.AI: https://www.youtube.com/watch?v=9d4jmPmTWmc

ML Mathematics:- 
Essence of linear algebra: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

NLU(NLP):- 
Stanford CS224U: Natural Language Understanding : https://www.youtube.com/watch?v=tZ_Jrc_nRJY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20
AWS contributes novel causal machine learning algorithms to DoWhy Python library: https://www.amazon.science/blog/aws-contributes-novel-causal-machine-learning-algorithms-to-dowhy
NLP-In-Hindi: https://github.com/krishnaik06/NLP-In-Hindi/blob/main/RoadMap%20Of%20NLP.pdf
Advanced Topic Modeling Tutorial: How to Use SVD & NMF in Python: https://hackernoon.com/advanced-topic-modeling-tutorial-how-to-use-svd-and-nmf-in-python-to-find-topics-in-text
Topic Modeling with NMF and SVD: https://nbviewer.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb#PyTorch

SQL:- 
CTE in SQL (Common Table Expression) | SQL WITH Clause | CTE Query Performance | Advanced SQL: https://www.youtube.com/watch?v=zg9GNdX-Q9g&list=PLtgiThe4j67osrX6iUEpo7J4Gkh_G25Y_&index=4

WEB3:- 
OTT Platform Architecture Netflix: https://whimsical.com/scaler-ott-platforms-Kei65tZhEKtJqkyWExbYNB
Distributed System Design Patterns: https://medium.com/@nishantparmar/distributed-system-design-patterns-2d20908fecfc
WEB3 Intro Course: https://www.youtube.com/playlist?list=PL5dTjWUk_cPZgnfsw3h7usMAa5W5bZzxq

WebPack5:- 
wp5-intro-video-code: https://github.com/jherr/wp5-intro-video-code/tree/master/packages/search

Snowflake:-
Snowflake Labs: https://github.com/Snowflake-Labs
Machine Learning with Snowpark Python: https://quickstarts.snowflake.com/guide/machine_learning_with_snowpark_python/index.html?index=..%2F..index#0
sfguide-citibike-ml-snowpark-python: https://github.com/Snowflake-Labs/sfguide-citibike-ml-snowpark-python
What’s New: Snowflake for Data Science: https://events.snowflake.com/snowday/americas/agenda/session/651839
Getting Started with Snowpark Python: https://quickstarts.snowflake.com/guide/getting_started_with_snowpark_python/index.html?index=..%2F..index#0


Airflow:-
airflow-dbt-demo: https://github.com/astronomer/airflow-dbt-demo
Deploying a Modern Cloud-Based Data Engineering Architecture w/ AWS, Airflow & Snowflake-Wes Sankey: https://www.youtube.com/watch?v=924V6edYW_w
Next Generation Big Data Pipelines with Prefect and Dask: https://www.youtube.com/watch?v=R6z77ZNJvho
tickit-data-lake-demo (Airflow Dag example to run Glue jobs , big data jobs etc.) : https://github.com/garystafford/tickit-data-lake-demo/tree/main/dags
Building a Data Lake on AWS with Apache Airflow: https://www.youtube.com/watch?v=RqjmC8iZEUo



Terraform:- 
Using Terraform to build a serverless Airflow via Amazon Managed Workflows and automatic DAG sync using GitHub Actions: https://blog.telsemeyer.com/2021/07/20/using-terraform-to-build-a-serverless-airflow-via-amazon-managed-workflows-and-automatic-dag-sync-using-github-actions/

Visual Subnet Calculator:- https://www.davidc.net/sites/default/subnets/subnets.html#
