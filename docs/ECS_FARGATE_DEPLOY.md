# ECS Fargate Deploy (API)

This repo’s backend API is a FastAPI app. Locally it is started as:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

In ECS Fargate we run the same entrypoint (no `--reload`), and pass a single `DATABASE_URL` that points at RDS.

## Why `DATABASE_URL` (and TLS) matters

- The backend and scripts already standardize on **`DATABASE_URL`**.
- Your `/health` endpoint reports DB connectivity but still returns HTTP 200 even if DB is unavailable/misconfigured, which is ideal for initial ECS bring-up.
- For RDS with `sslmode=verify-full`, libpq needs a CA bundle file.
  - The Docker image downloads the AWS global bundle into:
    - `/etc/ssl/certs/rds-global-bundle.pem`

So your production `DATABASE_URL` should include:

`... ?sslmode=verify-full&sslrootcert=/etc/ssl/certs/rds-global-bundle.pem`

## 1) Build & push image to ECR

From repo root:

```bash
aws ecr create-repository --repository-name friday-api

# Login to ECR
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

# Build (uses backend/Dockerfile but repo root as context)
docker build -f backend/Dockerfile -t friday-api:latest .

# Tag & push
docker tag friday-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/friday-api:<TAG>
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/friday-api:<TAG>
```

## 2) Store `DATABASE_URL` in Secrets Manager

Create a secret whose value is the full URL (recommended, simplest):

```bash
aws secretsmanager create-secret \
  --name friday/DATABASE_URL \
  --secret-string "postgresql://friday:<PASSWORD>@<RDS_ENDPOINT>:5432/friday?sslmode=verify-full&sslrootcert=/etc/ssl/certs/rds-global-bundle.pem"
```

## 3) ECS task definition

Use `ecs/fargate-task-definition.json` as a template:

- Replace:
  - `<ACCOUNT_ID>`, `<REGION>`, `<TAG>`
- Ensure the task execution role can read the secret:
  - `secretsmanager:GetSecretValue` on `friday/DATABASE_URL`
  - plus standard ECS execution permissions for ECR + CloudWatch Logs

Register it:

```bash
aws ecs register-task-definition --cli-input-json file://ecs/fargate-task-definition.json
```

## 4) Run a service behind an ALB

High-level wiring:

- **ALB target group**:
  - protocol: HTTP
  - port: 8000 (or map ALB listener 80/443 → target group 8000)
  - health check path: `/health`
- **Security groups**:
  - ALB SG allows inbound from internet (80/443)
  - ECS task SG allows inbound **from ALB SG** to container port 8000
  - RDS SG allows inbound **from ECS task SG** to 5432
- **Networking**:
  - tasks in private subnets (recommended)
  - NAT gateway or VPC endpoints for ECR/Logs/Secrets if no outbound internet

## 5) Validate

Once the service is running:

- Hit `/health` on the ALB.
- Expect a payload like:
  - `status: ok`
  - `db.ok: true` if RDS is reachable and `DATABASE_URL` is correct
  - `db.ok: false` (still HTTP 200) during initial bring-up if DB/security groups aren’t ready yet

