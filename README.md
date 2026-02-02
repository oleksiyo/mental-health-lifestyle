## Mental Health & Lifestyle – Risk Prediction Project

This project analyzes how lifestyle, work, and social factors are related to mental health issues and builds a machine learning model that **predicts whether a person is likely to have a mental health issue**.

The project is structured and documented to satisfy the course requirements (EDA, multiple models with tuning, scripts, web service, Docker, and deployment instructions).

![image](./images/logo.png)

---

## Problem Description

Modern lifestyle and work conditions strongly affect mental health, but it is often unclear **which concrete factors are most important** and **who is at higher risk**.  
In this project we use a survey-style dataset with demographic, work, lifestyle and mental-health–related questions to:

- **Predict** whether a person has a mental health issue (`Has_Mental_Health_Issue` – target variable)
- **Understand** which features (stress, sleep, workload, support, etc.) are the most important drivers
- **Provide a simple API** that can be used to obtain predictions for new records

### Target and Business Use

> **Goal**: Given a person’s characteristics and lifestyle indicators, predict the probability that they have a mental health issue.

Potential uses:

- **HR / People teams** – identify groups with higher risk and adjust workload, benefits, and support programs
- **Well-being products / apps** – provide early screening and recommend interventions
- **Researchers / analysts** – study associations between lifestyle and mental health

**Task type**: Binary classification  
**Target column**: `Has_Mental_Health_Issue` (0 = no, 1 = yes)  
**Main metric**: ROC AUC (plus accuracy, precision, recall)

**Models trained**: 
- Logistic Regression (baseline with tuning)
- Random Forest (with hyperparameter tuning)
- XGBoost (with hyperparameter tuning)

The best performing model based on validation ROC-AUC is automatically selected and saved.

---

## Dataset

- File: `data/mental_health.csv` (already committed to this repository)
- Each row represents one survey respondent.

### Column Examples

The dataset contains 50+ features. Some key groups:

- **Demographics**: `Age`, `Gender`, `Country`, `Education`, `Marital_Status`, `Income_Level`
- **Work & stress**:  
  `Employment_Status`, `Work_Hours_Per_Week`, `Remote_Work`,  
  `Job_Satisfaction`, `Work_Stress_Level`, `Work_Life_Balance`,  
  `Ever_Bullied_At_Work`, `Company_Mental_Health_Support`
- **Lifestyle**:  
  `Exercise_Per_Week`, `Sleep_Hours_Night`, `Caffeine_Drinks_Day`,  
  `Alcohol_Frequency`, `Smoking`, `Screen_Time_Hours_Day`,  
  `Social_Media_Hours_Day`, `Hobby_Time_Hours_Week`, `Diet_Quality`
- **Mental health indicators & history**:  
  `Feeling_Sad_Down`, `Loss_Of_Interest`, `Anxious_Nervous`,  
  `Panic_Attacks`, `Family_History_Mental_Illness`,  
  `Previously_Diagnosed`, `Ever_Sought_Treatment`, `On_Therapy_Now`,  
  `On_Medication`, `Trauma_History`, `Social_Support`, `Loneliness`,  
  `Discuss_Mental_Health`
- **Target**: `Has_Mental_Health_Issue` (0 or 1)

The data is small enough to work comfortably in a notebook, but large enough for meaningful modeling.

---

## Project Structure

```text
mental-health-lifestyle/
│
├── data/
│   └── mental_health.csv        # Dataset used in the project
│
├── notebook.ipynb              # EDA, feature analysis, and model development
│
├── train.py                    # Training script: trains multiple models (LR, RF, XGBoost), tuning, saves best
├── serve.py                    # Flask web service for predictions (/health, /predict)
├── predict.py                  # Standalone prediction script (CLI usage)
├── test_api.py                 # API testing script with examples
├── deploy.py                   # Deployment script for Docker, AWS, Railway
│
├── model.bin                   # Saved model + vectorizer (created after running train.py)
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container for the prediction service
└── README.md                   # This file
```

> Note: `model.bin` is created after you run `train.py` for the first time.

---

## EDA and Modeling (Summary)

The full analysis is in `notebook.ipynb`. Main steps:

- **Data preparation & cleaning**
  - Load `data/mental_health.csv`
  - Check missing values and handle them (simple imputation for numeric; separate category for missing in categoricals)
  - Cast `Has_Mental_Health_Issue` to integer 0/1
- **EDA**
  - Distribution of key numeric features: `Age`, `Work_Stress_Level`, `Sleep_Hours_Night`, `Work_Hours_Per_Week`, etc.
  - Analysis of target variable: class balance of `Has_Mental_Health_Issue`
  - Relationships between target and important drivers (e.g. stress level, sleep, loneliness, social support)
  - Correlation / feature importance using tree-based models

  ![image](./images/001.png)

- **Models & tuning**
  - Model 1: **Logistic Regression** with hyperparameter tuning (C, class_weight)
  - Model 2: **Random Forest** with hyperparameter tuning (n_estimators, max_depth, min_samples, etc.)
  - Model 3: **XGBoost** with hyperparameter tuning (learning_rate, max_depth, subsample, etc.)
  - All models use **RandomizedSearchCV** with 5-fold cross-validation
  - Model selection based on **ROC AUC** on validation data
  - The **best model** is automatically selected and saved to `model.bin` with DictVectorizer
![image](./images/002.png)
---

## How to Run Locally (without Docker)

### 1. Clone the repository

```bash
git clone git@github.com:oleksiyo/mental-health-lifestyle.git
cd mental-health-lifestyle
```

### 2. Create and activate virtual environment

On **Windows PowerShell**:

```bash
python -m venv venv
.\venv\Scripts\activate
```

On **macOS / Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

After success, you should see `model.bin` in the root folder.

### 5. Run the prediction web service

Start the Flask service:

```bash
python serve.py
```

By default, the service listens on `http://0.0.0.0:9696`.

#### Health check

Open in browser or with `curl`:

```bash
curl http://localhost:9696/health
```

Expected response:

```json
{ "status": "ok" }
```

#### Example prediction request

`POST /predict` expects a JSON object with a set of features used during training  
(demographic, work and lifestyle fields).

Example:

```bash
curl -X POST "http://localhost:9696/predict" ^
     -H "Content-Type: application/json" ^
     -d "{
           \"Age\": 32,
           \"Gender\": \"Female\",
           \"Country\": \"Germany\",
           \"Education\": \"Bachelor\",
           \"Marital_Status\": \"Single\",
           \"Income_Level\": \"Middle\",
           \"Employment_Status\": \"Full-time\",
           \"Work_Hours_Per_Week\": 40,
           \"Remote_Work\": \"Hybrid\",
           \"Job_Satisfaction\": 6,
           \"Work_Stress_Level\": 7,
           \"Work_Life_Balance\": 5,
           \"Exercise_Per_Week\": \"1-2 times\",
           \"Sleep_Hours_Night\": 7.0,
           \"Caffeine_Drinks_Day\": 2,
           \"Alcohol_Frequency\": \"Rarely\",
           \"Smoking\": \"Never\",
           \"Screen_Time_Hours_Day\": 6.0,
           \"Social_Support\": 7,
           \"Loneliness\": 3,
           \"Discuss_Mental_Health\": \"Sometimes\"
         }"
```

Example JSON response:

```json
{
  "prediction": 0,
  "probability": 0.31
}
```

Where:
- **`prediction`** – model prediction (1 = likely mental health issue, 0 = unlikely)
- **`probability`** – predicted probability for class `1`

### 6. Test the API

You can test the API with the provided test script:

```bash
python test_api.py
```

This will run health checks and test predictions with sample data.

---

## Docker

The project includes a `Dockerfile` so the service can be containerized.

### Build the image

From the project root:

```bash
docker build -t mental-health-api .
```

### Run the container

```bash
docker run -it --rm -p 9696:9696 mental-health-api
```

Now the same endpoints are available:

- Health: `http://localhost:9696/health`
- Prediction: `http://localhost:9696/predict`

You can test with `python test_api.py` or use the curl examples above.

---



## Cloud Deployment (AWS)

### Prerequisites

1. **AWS Account** - Create at [aws.amazon.com](https://aws.amazon.com/)
2. **AWS CLI** - Install and configure:
   ```bash
   # Install AWS CLI
   # Windows: Download from https://aws.amazon.com/cli/
   # Mac: brew install awscli
   # Linux: sudo apt install awscli
   
   # Configure with your credentials
   aws configure
   # Enter: AWS Access Key ID, Secret Access Key, Region (e.g., us-east-1)
   ```
3. **Docker** - Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

### Automated Deployment (Recommended)

Use the deployment script for automated deployment:

```bash
python deploy.py
```

**Menu options:**
- **Option 1** - Build Docker image locally
- **Option 2** - Run Docker container locally for testing
- **Option 3** - Deploy to AWS ECR (push image to registry)
- **Option 4** - Create ECS Task Definition
- **Option 5** - Create ECS Service (instructions)
- **Option 6** - **Full deployment** (all steps automatically)

**For first-time deployment, select Option 6** - it will:
1. Build Docker image
2. Push to AWS ECR
3. Create ECS Task Definition
4. Provide instructions for ECS Service setup

### Manual Deployment

#### Step 1: Push to AWS ECR

```bash
# Get your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-east-1"
REPO_NAME="mental-health-api"

# Create ECR repository
aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push
docker build -t $REPO_NAME .
docker tag $REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest
```

#### Step 2: Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name mental-health-cluster --region $AWS_REGION
```

#### Step 3: Create Task Definition

The deployment script creates `ecs-task-definition.json`. Register it:

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json --region $AWS_REGION
```

Or create manually with these specs:
- **CPU**: 256 (0.25 vCPU)
- **Memory**: 512 MB
- **Container Port**: 9696
- **Launch Type**: Fargate

#### Step 4: Create ECS Service

```bash
# Get your VPC and subnet IDs
VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)
SUBNET_ID=$(aws ec2 describe-subnets --query 'Subnets[0].SubnetId' --output text --region $AWS_REGION)

# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name mental-health-sg \
  --description "Security group for Mental Health API" \
  --vpc-id $VPC_ID \
  --region $AWS_REGION \
  --query 'GroupId' --output text)

# Allow inbound traffic on port 9696
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 9696 \
  --cidr 0.0.0.0/0 \
  --region $AWS_REGION

# Create ECS service
aws ecs create-service \
  --cluster mental-health-cluster \
  --service-name mental-health-service \
  --task-definition mental-health-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region $AWS_REGION
```

#### Step 5: Get Service URL

```bash
# List tasks
TASK_ARN=$(aws ecs list-tasks --cluster mental-health-cluster --region $AWS_REGION --query 'taskArns[0]' --output text)

# Get task details and public IP
aws ecs describe-tasks --cluster mental-health-cluster --tasks $TASK_ARN --region $AWS_REGION --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text | xargs -I {} aws ec2 describe-network-interfaces --network-interface-ids {} --region $AWS_REGION --query 'NetworkInterfaces[0].Association.PublicIp' --output text
```

Your API will be available at: `http://<PUBLIC_IP>:9696`

### Cost Estimation

**AWS Fargate costs** (us-east-1):
- 0.25 vCPU: ~$0.012/hour = $8.64/month
- 512 MB memory: ~$0.001/hour = $0.76/month
- **Total: ~$9.40/month** (if running 24/7)

**ECR Storage**: First 500 MB free, then $0.10/GB/month

---

### Testing Deployed Service

After deployment, test your service:

```bash
# Update API_URL in test_api.py to your AWS public IP
# For example: API_URL = "http://3.85.123.456:9696"

python test_api.py
```

Or use curl:

```bash
# Replace <PUBLIC_IP> with your ECS task's public IP
curl http://<PUBLIC_IP>:9696/health

curl -X POST "http://<PUBLIC_IP>:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 34,
    "Gender": "Male",
    "Work_Stress_Level": 8,
    "Sleep_Hours_Night": 5.0,
    "Social_Support": 2,
    "Loneliness": 8
  }'
```

**Expected response:**
```json
{
  "prediction": 1,
  "probability": 0.8234
}
```



## Cloud Deployment (Deploy to Kubernetes)

1. Build Docker image
```bash
docker build -t mental-health-lifestyle:latest .
```

2. Deploy to Kubernetes
```bash
kubectl apply -f k8s/
```

3. Check status
```bash
kubectl get pods
kubectl get svc
```

4. Get service URL (for minikube)
```bash
minikube service pmental-health-lifestyle --url
```

5. Test
```bash
curl http://<SERVICE-URL>/health
```

---

**Files:**

- k8s/deployment.yaml - Pod deployment
- k8s/service.yaml - LoadBalancer service on port 80



##  Next Steps & Future Improvements

- **Add more models**: Try CatBoost, LightGBM, or ensemble methods (Stacking, Voting)
- **Cross-validation**: Use StratifiedKFold for more robust validation
- **Input validation**: Add Pydantic models for request validation
- **Async support**: Migrate to FastAPI for better async handling
- **Kubernetes**: Deploy with Kubernetes for better scalability

---