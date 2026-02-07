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
# Auth config (non-secret)
############################################
COGNITO_DOMAIN="https://us-west-1b9vdzkuiu.auth.us-west-1.amazoncognito.com"
COGNITO_ISSUER="https://cognito-idp.us-west-1.amazonaws.com/us-west-1_b9VdZKUiu"
COGNITO_CLIENT_ID="35p0phf7tk0231pr5i4pippr4s"
COGNITO_REDIRECT_URI="https://api.fridayarchive.org/auth/oauth/cognito/callback"
UI_REDIRECT_AFTER_LOGIN="https://fridayarchive.org/"
COOKIE_DOMAIN=".fridayarchive.org"
COOKIE_SECURE="true"
SESSION_COOKIE_NAME="friday_session"
COGNITO_USER_POOL_ID=us-west-1_b9VdZKUiu

# Existing env var you already inject (kept)
S3_PDF_BUCKET_VALUE="${S3_PDF_BUCKET:-fridayarchive.org}"

############################################
# Secrets Manager (secret names you created)
############################################
INCLUDE_COGNITO_CLIENT_SECRET="true"

COGNITO_CLIENT_SECRET_NAME="cognito-client"
APP_SESSION_SECRET_NAME="app-session-secret"



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
# Preflight checks
############################################
command -v jq >/dev/null 2>&1 || { echo "ERROR: jq is required."; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI is required."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required."; exit 1; }

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
# 6) Pull task definition JSON, update image/env/secrets, register new revision
############################################
echo "Downloading current task definition JSON..."
TASKDEF_JSON="$(aws ecs describe-task-definition \
  --region "${AWS_REGION}" \
  --task-definition "${CURRENT_TASKDEF_ARN}")"

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

echo "Resolving Secrets Manager ARNs..."

APP_SESSION_SECRET_ARN="$(aws secretsmanager describe-secret \
  --region "${AWS_REGION}" \
  --secret-id "${APP_SESSION_SECRET_NAME}" \
  --query "ARN" --output text)"

if [[ -z "${APP_SESSION_SECRET_ARN}" || "${APP_SESSION_SECRET_ARN}" == "None" ]]; then
  echo "ERROR: Could not resolve ARN for Secrets Manager secret: ${APP_SESSION_SECRET_NAME}"
  exit 1
fi


COGNITO_CLIENT_SECRET_ARN=""
if [[ "${INCLUDE_COGNITO_CLIENT_SECRET}" == "true" ]]; then
  COGNITO_CLIENT_SECRET_ARN="$(aws secretsmanager describe-secret \
    --region "${AWS_REGION}" \
    --secret-id "${COGNITO_CLIENT_SECRET_NAME}" \
    --query "ARN" --output text || true)"

  if [[ -z "${COGNITO_CLIENT_SECRET_ARN}" || "${COGNITO_CLIENT_SECRET_ARN}" == "None" ]]; then
    echo "ERROR: INCLUDE_COGNITO_CLIENT_SECRET=true but could not resolve ARN for: ${COGNITO_CLIENT_SECRET_NAME}"
    echo "If you're not using OAuth right now, set INCLUDE_COGNITO_CLIENT_SECRET=false."
    exit 1
  fi
fi

echo "APP_SESSION_SECRET_ARN: ${APP_SESSION_SECRET_ARN}"
if [[ "${INCLUDE_COGNITO_CLIENT_SECRET}" == "true" ]]; then
  echo "COGNITO_CLIENT_SECRET_ARN: ${COGNITO_CLIENT_SECRET_ARN}"
else
  echo "COGNITO_CLIENT_SECRET: (not injected; INCLUDE_COGNITO_CLIENT_SECRET=false)"
fi

echo "Updating task definition object (image, env, secrets)..."

UPDATED_TASKDEF_OBJ="$(echo "${TASKDEF_OBJ}" | jq \
  --arg img "${IMAGE_SHA}" \
  --argjson idx "${CONTAINER_INDEX}" \
  --arg s3bucket "${S3_PDF_BUCKET_VALUE}" \
  --arg cognito_domain "${COGNITO_DOMAIN}" \
  --arg cognito_issuer "${COGNITO_ISSUER}" \
  --arg cognito_client_id "${COGNITO_CLIENT_ID}" \
  --arg cognito_redirect_uri "${COGNITO_REDIRECT_URI}" \
  --arg ui_redirect "${UI_REDIRECT_AFTER_LOGIN}" \
  --arg cookie_domain "${COOKIE_DOMAIN}" \
  --arg cookie_secure "${COOKIE_SECURE}" \
  --arg session_cookie_name "${SESSION_COOKIE_NAME}" \
  --arg app_session_secret_arn "${APP_SESSION_SECRET_ARN}" \
  --arg cognito_client_secret_arn "${COGNITO_CLIENT_SECRET_ARN}" \
  --arg include_cognito_secret "${INCLUDE_COGNITO_CLIENT_SECRET}" \
'
  .containerDefinitions[$idx].image = $img

  # --- Environment vars (non-secret) ---
  | .containerDefinitions[$idx].environment |= (
      ( . // [] )
      | map(select(
          .name != "S3_PDF_BUCKET" and
          .name != "COGNITO_DOMAIN" and
          .name != "COGNITO_ISSUER" and
          .name != "COGNITO_CLIENT_ID" and
          .name != "COGNITO_REDIRECT_URI" and
          .name != "UI_REDIRECT_AFTER_LOGIN" and
          .name != "COOKIE_DOMAIN" and
          .name != "COOKIE_SECURE" and
          .name != "SESSION_COOKIE_NAME"
        ))
      + [
          {"name":"S3_PDF_BUCKET","value":$s3bucket},
          {"name":"COGNITO_DOMAIN","value":$cognito_domain},
          {"name":"COGNITO_ISSUER","value":$cognito_issuer},
          {"name":"COGNITO_CLIENT_ID","value":$cognito_client_id},
          {"name":"COGNITO_REDIRECT_URI","value":$cognito_redirect_uri},
          {"name":"UI_REDIRECT_AFTER_LOGIN","value":$ui_redirect},
          {"name":"COOKIE_DOMAIN","value":$cookie_domain},
          {"name":"COOKIE_SECURE","value":$cookie_secure},
          {"name":"SESSION_COOKIE_NAME","value":$session_cookie_name}
        ]
    )

  # --- Secrets (Secrets Manager references) ---
  | .containerDefinitions[$idx].secrets |= (
      ( . // [] )
      # remove prior injected entries to avoid duplicates
      | map(select(
          .name != "APP_SESSION_SECRET" and
          .name != "COGNITO_CLIENT_SECRET"
        ))
      + (
          if $include_cognito_secret == "true" then
            [
              {"name":"APP_SESSION_SECRET","valueFrom":($app_session_secret_arn + ":APP_SESSION_SECRET::")},
              {"name":"COGNITO_CLIENT_SECRET","valueFrom":($cognito_client_secret_arn + ":COGNITO_CLIENT_SECRET::")}
            ]
          else
            [
              {"name":"APP_SESSION_SECRET","valueFrom":($app_session_secret_arn + ":APP_SESSION_SECRET::")}
            ]
          end
        )
    )
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

# Verify the new task def has the image we expect (catches jq/script bugs)
ACTUAL_IMAGE="$(aws ecs describe-task-definition \
  --region "${AWS_REGION}" \
  --task-definition "${NEW_TASKDEF_ARN}" \
  --query "taskDefinition.containerDefinitions[${CONTAINER_INDEX}].image" \
  --output text)"
echo "Image in new task definition: ${ACTUAL_IMAGE}"
if [[ "${ACTUAL_IMAGE}" != "${IMAGE_SHA}" ]]; then
  echo "ERROR: Task definition image mismatch. Expected: ${IMAGE_SHA}"
  exit 1
fi

############################################
# 7) Update ECS service to the new task def and wait until stable
############################################
echo "Updating ECS service (force new deployment)..."
aws ecs update-service \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --task-definition "${NEW_TASKDEF_ARN}" \
  --force-new-deployment \
  --output text \
  --query "service.deployments"

echo "Waiting for service to become stable..."
aws ecs wait services-stable \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}"

############################################
# 8) Verify /health shows the deployed SHA (with retries)
############################################
echo "Verifying health endpoint (will retry for ~5 minutes)..."

MAX_ATTEMPTS=60
SLEEP_SECONDS=5
HEALTH=""

for i in $(seq 1 ${MAX_ATTEMPTS}); do
  if HEALTH="$(curl -fsS "${HEALTHCHECK_URL}" 2>/dev/null)"; then
    BUILD="$(echo "${HEALTH}" | jq -r '.build // empty' || true)"

    if [[ "${BUILD}" == "${SHA}" ]]; then
      echo "✅ Deploy complete. Running build SHA: ${BUILD}"
      exit 0
    else
      echo "SHA mismatch (attempt ${i}/${MAX_ATTEMPTS}): expected ${SHA}, got ${BUILD}. Retrying in ${SLEEP_SECONDS}s..."
    fi
  else
    echo "Health not ready yet (attempt ${i}/${MAX_ATTEMPTS}). Sleeping ${SLEEP_SECONDS}s..."
  fi

  sleep ${SLEEP_SECONDS}
done

echo "ERROR: Health check failed after ${MAX_ATTEMPTS} attempts."
echo "Expected SHA: ${SHA}"
echo "Last response: ${HEALTH:-<none>}"
echo ""
echo "If the API is still returning the old SHA, check in AWS ECS:"
echo "  - Service ${ECS_SERVICE} → Events (did the new deployment succeed?)"
echo "  - Service → Tasks: are new tasks (task def with image :${SHA}) running and healthy?"
echo "  - Stopped tasks: any exit code / reason showing why new tasks failed?"
exit 1
