#!/usr/bin/env bash
# Portability experiment (D-028): the SAME inprocess image on Google Cloud Run, for comparison with Render.
# Needs: gcloud logged in, a project with billing + run/artifactregistry APIs enabled (docs/SETUP.md §5.1c),
# Docker, and a published artifact release (make publish-artifacts).
#   ./deploy/cloudrun.sh <PROJECT_ID> [REGION]           → prints the service URL
set -euo pipefail
PROJECT=${1:?usage: deploy/cloudrun.sh <PROJECT_ID> [REGION]}
REGION=${2:-asia-south1}                                   # Mumbai
REPO=finsight
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/finsight-api:$(git rev-parse --short HEAD)"
ARTIFACTS_URL=${ARTIFACTS_URL:-https://github.com/bharathakn/FinSight-AI/releases/latest/download/finsight-artifacts.tar.gz}
CORS=${FINSIGHT_CORS_ORIGINS:-http://localhost:5173}
cd "$(dirname "$0")/.."

gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION" --project "$PROJECT"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -f docker/Dockerfile --build-arg SERVICE=gateway --build-arg REQS=full \
  --build-arg ARTIFACTS_URL="$ARTIFACTS_URL" -t "$IMAGE" .
docker push "$IMAGE"

# scale-to-zero (min 0), small ceiling (max 2) keeps the demo inside the free tier; 512 MiB = Render free size
gcloud run deploy finsight-api --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --platform managed --allow-unauthenticated --port 8000 --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 2 --concurrency 20 --timeout 60 \
  --set-env-vars "^@^FINSIGHT_SERVICE_MODE=inprocess@FINSIGHT_CORS_ORIGINS=${CORS}"   # ^@^ = custom delimiter (CORS list has commas)
gcloud run services describe finsight-api --project "$PROJECT" --region "$REGION" --format 'value(status.url)'
