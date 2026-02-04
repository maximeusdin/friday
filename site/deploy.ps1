# Deploy Friday Research Console to S3
# Usage: .\deploy.ps1 -BucketName <bucket-name> [-CloudFrontDistId <distribution-id>]

param(
    [Parameter(Mandatory=$true)]
    [string]$BucketName,
    
    [Parameter(Mandatory=$false)]
    [string]$CloudFrontDistId
)

$ErrorActionPreference = "Stop"

Write-Host "======================================"
Write-Host "Friday Research Console - S3 Deployment"
Write-Host "======================================"
Write-Host ""
Write-Host "Bucket: $BucketName"
if ($CloudFrontDistId) {
    Write-Host "CloudFront: $CloudFrontDistId"
}
Write-Host ""

# Check if out/ directory exists
if (-not (Test-Path "out")) {
    Write-Host "Error: 'out/' directory not found." -ForegroundColor Red
    Write-Host "Run 'npm run build' first to generate the static site."
    exit 1
}

# Confirm deployment
$confirm = Read-Host "Deploy to s3://$BucketName? (y/N)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Deployment cancelled."
    exit 0
}

Write-Host ""
Write-Host "Uploading files to S3..."

# Sync all files
aws s3 sync out/ "s3://$BucketName" --delete

Write-Host ""
Write-Host "Setting cache headers..."

# Long cache for static assets
aws s3 cp "s3://$BucketName" "s3://$BucketName" `
    --recursive `
    --exclude "*" `
    --include "*.js" `
    --include "*.css" `
    --include "*.woff" `
    --include "*.woff2" `
    --include "*.png" `
    --include "*.jpg" `
    --include "*.svg" `
    --include "*.ico" `
    --cache-control "public, max-age=31536000, immutable" `
    --metadata-directive REPLACE

# Short cache for HTML
aws s3 cp "s3://$BucketName" "s3://$BucketName" `
    --recursive `
    --exclude "*" `
    --include "*.html" `
    --cache-control "public, max-age=0, must-revalidate" `
    --metadata-directive REPLACE

# Invalidate CloudFront cache if specified
if ($CloudFrontDistId) {
    Write-Host ""
    Write-Host "Invalidating CloudFront cache..."
    aws cloudfront create-invalidation `
        --distribution-id $CloudFrontDistId `
        --paths "/*"
}

Write-Host ""
Write-Host "======================================"
Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host "======================================"
Write-Host ""
Write-Host "Site URL: http://$BucketName.s3-website-us-east-1.amazonaws.com"
if ($CloudFrontDistId) {
    Write-Host "CloudFront will update in a few minutes."
}
Write-Host ""
