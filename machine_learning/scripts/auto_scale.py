import boto3

from config import EndpointDeploymentConfig

client = boto3.client(
    "application-autoscaling"
)  # Common class representing Application Auto Scaling for SageMaker amongst other services

resource_id = (
    "endpoint/" + EndpointDeploymentConfig.endpoint_name + "/variant/AllTraffic"
)  # This is the format in which application autoscaling references the endpoint

# Configure Autoscaling on asynchronous endpoint down to zero instances
response = client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0,
    MaxCapacity=1,
)

response = client.put_scaling_policy(
    PolicyName="Invocations-ScalingPolicy",
    ServiceNamespace="sagemaker",  # The namespace of the AWS service that provides the resource.
    ResourceId=resource_id,  # Endpoint name
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
    PolicyType="TargetTrackingScaling",  # 'StepScaling'|'TargetTrackingScaling'
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 5,  # The target value for the metric. - here the metric is - SageMakerVariantInvocationsPerInstance
        "CustomizedMetricSpecification": {
            "MetricName": "ApproximateBacklogSizePerInstance", # https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-monitor.html
            "Namespace": "AWS/SageMaker",
            "Dimensions": [{"Name": "EndpointName", "Value": EndpointDeploymentConfig.endpoint_name}],
            "Statistic": "Average",
        },
        "ScaleInCooldown": 600,  # The cooldown period helps you prevent your Auto Scaling group from launching or terminating
        # additional instances before the effects of previous activities are visible.
        # You can configure the length of time based on your instance startup time or other application needs.
        # ScaleInCooldown - The amount of time, in seconds, after a scale in activity completes before another scale in activity can start.
        "ScaleOutCooldown": 300  # ScaleOutCooldown - The amount of time, in seconds, after a scale out activity completes before another scale out activity can start.
        # 'DisableScaleIn': True|False - ndicates whether scale in by the target tracking policy is disabled.
        # If the value is true , scale in is disabled and the target tracking policy won't remove capacity from the scalable resource.
    },
)