"""
AWS Deployment script for Mental Health API
Deploys Docker container to AWS ECR and ECS
"""
import os
import sys
import subprocess
import json


def check_aws_cli():
    """Check if AWS CLI is installed and configured"""
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_docker_image(tag="mental-health-api:latest"):
    """Build Docker image"""
    print(f"Building Docker image: {tag}")
    print("=" * 60)
    
    result = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode == 0:
        print(f"\n✓ Successfully built image: {tag}")
        return True
    else:
        print(f"\n✗ Failed to build image")
        return False


def run_local_docker(tag="mental-health-api:latest", port=9696):
    """Run Docker container locally"""
    print(f"Running Docker container on port {port}")
    print("=" * 60)
    
    cmd = [
        "docker", "run",
        "-it", "--rm",
        "-p", f"{port}:9696",
        tag
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop the container\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nContainer stopped")


def deploy_to_aws_ecr(region="us-east-1", repo_name="mental-health-api"):
    """
    Deploy to AWS Elastic Container Registry (ECR)
    Prerequisites:
    - AWS CLI installed and configured
    - Docker installed
    """
    print("Deploying to AWS ECR")
    print("=" * 60)
    
    # Get AWS account ID
    result = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print("✗ AWS CLI not configured. Run: aws configure")
        return False
    
    account_id = result.stdout.strip()
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}"
    
    print(f"AWS Account ID: {account_id}")
    print(f"ECR Repository: {ecr_uri}")
    print()
    
    # Step 1: Create ECR repository (if doesn't exist)
    print("Step 1: Creating ECR repository...")
    subprocess.run([
        "aws", "ecr", "create-repository",
        "--repository-name", repo_name,
        "--region", region
    ], capture_output=True)
    
    # Step 2: Authenticate Docker to ECR
    print("Step 2: Authenticating Docker to ECR...")
    result = subprocess.run([
        "aws", "ecr", "get-login-password",
        "--region", region
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("✗ Failed to get ECR login password")
        return False
    
    password = result.stdout.strip()
    subprocess.run([
        "docker", "login",
        "--username", "AWS",
        "--password-stdin",
        f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    ], input=password, text=True)
    
    # Step 3: Build image
    print("Step 3: Building Docker image...")
    if not build_docker_image(tag=ecr_uri):
        return False
    
    # Step 4: Push to ECR
    print("Step 4: Pushing image to ECR...")
    result = subprocess.run(["docker", "push", ecr_uri])
    
    if result.returncode == 0:
        print(f"\n✓ Successfully pushed to ECR: {ecr_uri}")
        print("\n" + "=" * 60)
        print("Next steps to deploy to ECS:")
        print("=" * 60)
        print(f"\n1. Create ECS Cluster:")
        print(f"   aws ecs create-cluster --cluster-name mental-health-cluster --region {region}")
        print(f"\n2. Create Task Definition:")
        print(f"   python deploy.py  # Select option 4 for automatic task definition")
        print(f"\n3. Create ECS Service:")
        print(f"   python deploy.py  # Select option 5 to create service with load balancer")
        print(f"\nImage URI: {ecr_uri}")
        return ecr_uri
    else:
        print("\n✗ Failed to push to ECR")
        return None


def create_ecs_task_definition(ecr_uri, region="us-east-1"):
    """Create ECS task definition"""
    print("\nCreating ECS Task Definition")
    print("=" * 60)
    
    task_def = {
        "family": "mental-health-api",
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "256",
        "memory": "512",
        "executionRoleArn": f"arn:aws:iam::{get_aws_account_id()}:role/ecsTaskExecutionRole",
        "containerDefinitions": [
            {
                "name": "mental-health-api",
                "image": ecr_uri,
                "portMappings": [
                    {
                        "containerPort": 9696,
                        "protocol": "tcp"
                    }
                ],
                "essential": True,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/mental-health-api",
                        "awslogs-region": region,
                        "awslogs-stream-prefix": "ecs"
                    }
                }
            }
        ]
    }
    
    # Create CloudWatch log group
    print("Creating CloudWatch log group...")
    subprocess.run([
        "aws", "logs", "create-log-group",
        "--log-group-name", "/ecs/mental-health-api",
        "--region", region
    ], capture_output=True)
    
    # Save to file
    task_def_file = "ecs-task-definition.json"
    with open(task_def_file, "w") as f:
        json.dump(task_def, f, indent=2)
    
    print(f"Task definition saved to: {task_def_file}")
    
    # Register task definition
    print("Registering task definition...")
    result = subprocess.run([
        "aws", "ecs", "register-task-definition",
        "--cli-input-json", f"file://{task_def_file}",
        "--region", region
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Task definition created successfully")
        print("\nNext: Create ECS service (option 5)")
        return True
    else:
        print(f"✗ Failed to create task definition: {result.stderr}")
        return False


def create_ecs_service(region="us-east-1"):
    """Create ECS service with Application Load Balancer"""
    print("\nCreating ECS Service")
    print("=" * 60)
    
    cluster_name = "mental-health-cluster"
    service_name = "mental-health-service"
    
    # Check if cluster exists
    result = subprocess.run([
        "aws", "ecs", "describe-clusters",
        "--clusters", cluster_name,
        "--region", region
    ], capture_output=True, text=True)
    
    if "ACTIVE" not in result.stdout:
        print(f"Creating ECS cluster: {cluster_name}")
        subprocess.run([
            "aws", "ecs", "create-cluster",
            "--cluster-name", cluster_name,
            "--region", region
        ])
    
    print(f"\n⚠️  Manual steps required:")
    print(f"\n1. Get your VPC ID and Subnets:")
    print(f"   aws ec2 describe-vpcs --region {region}")
    print(f"   aws ec2 describe-subnets --region {region}")
    
    print(f"\n2. Create security group allowing port 9696:")
    print(f"   aws ec2 create-security-group \\")
    print(f"     --group-name mental-health-sg \\")
    print(f"     --description 'Security group for Mental Health API' \\")
    print(f"     --vpc-id <YOUR_VPC_ID> --region {region}")
    print(f"   aws ec2 authorize-security-group-ingress \\")
    print(f"     --group-id <SECURITY_GROUP_ID> \\")
    print(f"     --protocol tcp --port 9696 --cidr 0.0.0.0/0")
    
    print(f"\n3. Create ECS service:")
    print(f"   aws ecs create-service \\")
    print(f"     --cluster {cluster_name} \\")
    print(f"     --service-name {service_name} \\")
    print(f"     --task-definition mental-health-api \\")
    print(f"     --desired-count 1 \\")
    print(f"     --launch-type FARGATE \\")
    print(f"     --network-configuration 'awsvpcConfiguration={{subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}}' \\")
    print(f"     --region {region}")
    
    print(f"\n4. Get public IP:")
    print(f"   aws ecs list-tasks --cluster {cluster_name} --region {region}")
    print(f"   aws ecs describe-tasks --cluster {cluster_name} --tasks <TASK_ARN> --region {region}")
    
    print("\n✓ Instructions displayed. Follow the manual steps above.")
    return True


def get_aws_account_id():
    """Get AWS account ID"""
    result = subprocess.run([
        "aws", "sts", "get-caller-identity",
        "--query", "Account",
        "--output", "text"
    ], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def deploy_full_stack(region="us-east-1"):
    """Full deployment workflow"""
    print("\n" + "=" * 60)
    print("Full AWS Deployment Workflow")
    print("=" * 60)
    
    # Step 1: Deploy to ECR
    print("\nStep 1: Deploying to AWS ECR...")
    ecr_uri = deploy_to_aws_ecr(region=region, repo_name="mental-health-api")
    if not ecr_uri:
        return False
    
    # Step 2: Create task definition
    print("\nStep 2: Creating ECS Task Definition...")
    if not create_ecs_task_definition(ecr_uri, region):
        return False
    
    # Step 3: Instructions for service
    print("\nStep 3: ECS Service Setup...")
    create_ecs_service(region)
    
    return True


def show_menu():
    """Show deployment options menu"""
    print("\n" + "=" * 60)
    print("Mental Health API - AWS Deployment")
    print("=" * 60)
    print("\nDeployment Options:")
    print("1. Build Docker image (local)")
    print("2. Run Docker container (local)")
    print("3. Deploy to AWS ECR")
    print("4. Create ECS Task Definition")
    print("5. Create ECS Service (manual steps)")
    print("6. Full AWS Deployment (all steps)")
    print("0. Exit")
    print()


def main():
    """Main deployment flow"""
    
    print("=" * 60)
    print("Mental Health API - AWS Deployment Script")
    print("=" * 60)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    if not check_aws_cli():
        print("✗ AWS CLI not installed or not configured")
        print("\nInstallation:")
        print("  Windows: https://aws.amazon.com/cli/")
        print("  Mac: brew install awscli")
        print("  Linux: sudo apt install awscli")
        print("\nConfiguration:")
        print("  aws configure")
        print("  Enter your AWS Access Key ID, Secret Access Key, and region")
        return
    else:
        print("✓ AWS CLI configured")
    
    # Get region
    region = input("\nEnter AWS region (default: us-east-1): ").strip() or "us-east-1"
    
    while True:
        show_menu()
        choice = input("Select option (0-6): ").strip()
        
        if choice == "0":
            print("Exiting...")
            break
        elif choice == "1":
            build_docker_image()
        elif choice == "2":
            run_local_docker()
        elif choice == "3":
            deploy_to_aws_ecr(region=region)
        elif choice == "4":
            ecr_uri = input("Enter ECR image URI: ").strip()
            if ecr_uri:
                create_ecs_task_definition(ecr_uri, region)
        elif choice == "5":
            create_ecs_service(region)
        elif choice == "6":
            deploy_full_stack(region)
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
