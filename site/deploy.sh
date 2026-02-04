#!/bin/bash
# Deploy Friday Research Console to S3
# Usage: ./deploy.sh <bucket-name> [--cloudfront <distribution-id>]

set -e

BUCKET_NAME=$1
CLOUDFRONT_DIST=""

# Parse arguments
shift
while [[ $# -gt 0 ]]; do
  case $1 in
    --cloudfront)
      CLOUDFRONT_DIST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$BUCKET_NAME" ]; then
  echo "Usage: ./deploy.sh <bucket-name> [--cloudfront <distribution-id>]"
  echo ""
  echo "Example:"
  echo "  ./deploy.sh friday-site-prod"
  echo "  ./deploy.sh friday-site-prod --cloudfront E1234567890ABC"
  exit 1
fi

echo "======================================"
echo "Friday Research Console - S3 Deployment"
echo "======================================"
echo ""
echo "Bucket: $BUCKET_NAME"
[ -n "$CLOUDFRONT_DIST" ] && echo "CloudFront: $CLOUDFRONT_DIST"
echo ""

# Check if out/ directory exists
if [ ! -d "out" ]; then
  echo "Error: 'out/' directory not found."
  echo "Run 'npm run build' first to generate the static site."
  exit 1
fi

# Confirm deployment
read -p "Deploy to s3://$BUCKET_NAME? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
  echo "Deployment cancelled."
  exit 0
fi

echo ""
echo "Uploading files to S3..."

# Sync all files
aws s3 sync out/ "s3://$BUCKET_NAME" --delete

echo ""
echo "Setting cache headers..."

# Long cache for static assets (JS, CSS, images)
aws s3 cp "s3://$BUCKET_NAME" "s3://$BUCKET_NAME" \
  --recursive \
  --exclude "*" \
  --include "*.js" \
  --include "*.css" \
  --include "*.woff" \
  --include "*.woff2" \
  --include "*.png" \
  --include "*.jpg" \
  --include "*.svg" \
  --include "*.ico" \
  --cache-control "public, max-age=31536000, immutable" \
  --metadata-directive REPLACE

# Short cache for HTML (for updates)
aws s3 cp "s3://$BUCKET_NAME" "s3://$BUCKET_NAME" \
  --recursive \
  --exclude "*" \
  --include "*.html" \
  --cache-control "public, max-age=0, must-revalidate" \
  --metadata-directive REPLACE

# Invalidate CloudFront cache if specified
if [ -n "$CLOUDFRONT_DIST" ]; then
  echo ""
  echo "Invalidating CloudFront cache..."
  aws cloudfront create-invalidation \
    --distribution-id "$CLOUDFRONT_DIST" \
    --paths "/*"
fi

echo ""
echo "======================================"
echo "Deployment complete!"
echo "======================================"
echo ""
echo "Site URL: http://$BUCKET_NAME.s3-website-us-east-1.amazonaws.com"
[ -n "$CLOUDFRONT_DIST" ] && echo "CloudFront will update in a few minutes."
echo ""
