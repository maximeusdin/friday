#!/usr/bin/env bash
set -euo pipefail

############################################
# CONFIG (edit these)
############################################
AWS_REGION="us-west-1"

# Your AWS account ID
AWS_ACCOUNT_ID="682405977227"

# ECR repo name
ECR_REPO_NAME="friday-api"

# ECS identifiers
ECS_CLUSTER="friday-cluster"
ECS_SERVICE="friday-api-service-fhkb95na"

# (Optional) task family override; if empty we auto-detect from service
TASK_FAMILY=""

# If your task definition has multiple containers and you want a specific one:
# Leave empty to update the first container definition.
CONTAINER_NAME=""

# Health endpoint to verify after deploy
HEALTHCHECK_URL="https://api.fridayarchive.org/health"

############################################
# Derived vars
############################################
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE_BASE="${ECR_REGISTRY}/${ECR_REPO_NAME}"

# Git SHA tag (short)
SHA="$(git rev-parse --short HEAD)"
IMAGE_SHA="${ECR_IMAGE_BASE}:${SHA}"

echo "Deploying SHA: ${SHA}"
echo "Target image: ${IMAGE_SHA}"

############################################
# 1) Build image with BUILD_SHA baked in
############################################
echo "Building Docker image..."
docker build \
  --build-arg BUILD_SHA="${SHA}" \
  --no-cache \
  -t "${ECR_REPO_NAME}:${SHA}" \
  -t "${ECR_REPO_NAME}:latest" \
  .

############################################
# 2) Login to ECR
############################################
echo "Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

############################################
# 3) Ensure ECR repo exists (safe if it already exists)
############################################
echo "Ensuring ECR repo exists..."
aws ecr describe-repositories --region "${AWS_REGION}" --repository-names "${ECR_REPO_NAME}" >/dev/null 2>&1 \
  || aws ecr create-repository --region "${AWS_REGION}" --repository-name "${ECR_REPO_NAME}" >/dev/null

############################################
# 4) Tag + push
############################################
echo "Tagging images..."
docker tag "${ECR_REPO_NAME}:${SHA}" "${IMAGE_SHA}"
docker tag "${ECR_REPO_NAME}:latest" "${ECR_IMAGE_BASE}:latest"

echo "Pushing images..."
docker push "${IMAGE_SHA}"
docker push "${ECR_IMAGE_BASE}:latest"

############################################
# 5) Discover current task definition from the service
############################################
echo "Fetching current service task definition..."
CURRENT_TASKDEF_ARN="$(aws ecs describe-services \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --query "services[0].taskDefinition" \
  --output text)"

if [[ "${CURRENT_TASKDEF_ARN}" == "None" || -z "${CURRENT_TASKDEF_ARN}" ]]; then
  echo "ERROR: Could not find current task definition for service."
  exit 1
fi

echo "Current task definition ARN: ${CURRENT_TASKDEF_ARN}"

############################################
# 6) Pull task definition JSON and create a new revision with updated image
############################################
echo "Downloading current task definition JSON..."
TASKDEF_JSON="$(aws ecs describe-task-definition \
  --region "${AWS_REGION}" \
  --task-definition "${CURRENT_TASKDEF_ARN}")"

# Extract the taskDefinition object
TASKDEF_OBJ="$(echo "${TASKDEF_JSON}" | jq '.taskDefinition')"

# Determine container index to update
if [[ -n "${CONTAINER_NAME}" ]]; then
  CONTAINER_INDEX="$(echo "${TASKDEF_OBJ}" | jq -r --arg name "${CONTAINER_NAME}" '
    .containerDefinitions | to_entries[] | select(.value.name==$name) | .key
  ' | head -n 1)"
  if [[ -z "${CONTAINER_INDEX}" || "${CONTAINER_INDEX}" == "null" ]]; then
    echo "ERROR: Could not find container named '${CONTAINER_NAME}' in task definition."
    exit 1
  fi
else
  CONTAINER_INDEX="0"
fi

echo "Updating container index: ${CONTAINER_INDEX}"

# Update the image field
UPDATED_TASKDEF_OBJ="$(echo "${TASKDEF_OBJ}" | jq --arg img "${IMAGE_SHA}" --argjson idx "${CONTAINER_INDEX}" '
  .containerDefinitions[$idx].image = $img
')"

# IMPORTANT: remove fields that register-task-definition does not accept
SANITIZED_TASKDEF_OBJ="$(echo "${UPDATED_TASKDEF_OBJ}" | jq '
  del(
    .taskDefinitionArn,
    .revision,
    .status,
    .requiresAttributes,
    .compatibilities,
    .registeredAt,
    .registeredBy
  )
')"

# Optional: override family if specified
if [[ -n "${TASK_FAMILY}" ]]; then
  SANITIZED_TASKDEF_OBJ="$(echo "${SANITIZED_TASKDEF_OBJ}" | jq --arg fam "${TASK_FAMILY}" '.family = $fam')"
fi

echo "Registering new task definition revision..."
NEW_TASKDEF_ARN="$(aws ecs register-task-definition \
  --region "${AWS_REGION}" \
  --cli-input-json "$(echo "${SANITIZED_TASKDEF_OBJ}")" \
  --query "taskDefinition.taskDefinitionArn" \
  --output text)"

echo "New task definition ARN: ${NEW_TASKDEF_ARN}"

############################################
# 7) Update ECS service to the new task def and wait until stable
############################################
echo "Updating ECS service..."
aws ecs update-service \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --task-definition "${NEW_TASKDEF_ARN}" \
  >/dev/null

echo "Waiting for service to become stable..."
aws ecs wait services-stable \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}"

############################################
# 8) Verify /health shows the deployed SHA (with retries)
############################################
echo "Verifying health endpoint (will retry for ~2.5 minutes)..."

MAX_ATTEMPTS=30
SLEEP_SECONDS=5

for i in $(seq 1 ${MAX_ATTEMPTS}); do
  if HEALTH="$(curl -fsS "${HEALTHCHECK_URL}" 2>/dev/null)"; then
    echo "Health response: ${HEALTH}"
    
    # Check build SHA in response (expects JSON field "build")
    BUILD="$(echo "${HEALTH}" | jq -r '.build // empty' || true)"
    
    if [[ "${BUILD}" == "${SHA}" ]]; then
      echo "✅ Deploy complete. Running build SHA: ${BUILD}"
      exit 0
    else
      echo "SHA mismatch (attempt ${i}/${MAX_ATTEMPTS}): expected ${SHA}, got ${BUILD}"
      echo "Old container may still be draining. Retrying in ${SLEEP_SECONDS}s..."
    fi
  else
    echo "Health not ready yet (attempt ${i}/${MAX_ATTEMPTS}). Sleeping ${SLEEP_SECONDS}s..."
  fi
  
  sleep ${SLEEP_SECONDS}
done

echo "ERROR: Health check failed after ${MAX_ATTEMPTS} attempts."
echo "Expected SHA: ${SHA}"
echo "Last response: ${HEALTH:-<none>}"
exit 1
