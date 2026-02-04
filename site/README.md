# Friday Research Console - S3 Static Site Deployment

This directory contains the frontend static site ready for deployment to AWS S3.

## Prerequisites

- Node.js 18+ (check `.nvmrc`)
- AWS CLI configured with appropriate credentials
- An S3 bucket configured for static website hosting
- (Optional) CloudFront distribution for HTTPS and caching
- Backend API deployed (ECS, Lambda, or other compute service)

## Project Structure

```
site/
├── src/
│   ├── app/           # Next.js App Router pages
│   ├── components/    # React components
│   ├── lib/           # API client and utilities
│   └── types/         # TypeScript type definitions
├── public/            # Static assets (favicon, etc.)
├── next.config.js     # Next.js config (static export)
├── package.json       # Dependencies and scripts
└── README.md          # This file
```

## Build Instructions

### 1. Install Dependencies

```bash
cd site
npm install
```

### 2. Configure API URL

Set the `NEXT_PUBLIC_API_URL` environment variable to your deployed backend API URL:

```bash
# For local development (if backend runs on localhost:8000)
export NEXT_PUBLIC_API_URL=http://localhost:8000/api

# For production (replace with your actual API URL)
export NEXT_PUBLIC_API_URL=https://api.friday.example.com/api
```

You can also create a `.env.local` file:

```
NEXT_PUBLIC_API_URL=https://api.friday.example.com/api
```

### 3. Build Static Site

```bash
npm run build
```

This generates the static site in the `out/` directory.

### 4. Preview Locally (Optional)

```bash
npx serve out
```

## Deploy to S3

### Option A: AWS CLI

```bash
# Sync to S3 bucket
aws s3 sync out/ s3://YOUR-BUCKET-NAME --delete

# Set cache headers for assets
aws s3 cp s3://YOUR-BUCKET-NAME s3://YOUR-BUCKET-NAME \
  --recursive \
  --exclude "*" \
  --include "*.js" \
  --include "*.css" \
  --cache-control "public, max-age=31536000, immutable" \
  --metadata-directive REPLACE

# Set shorter cache for HTML
aws s3 cp s3://YOUR-BUCKET-NAME s3://YOUR-BUCKET-NAME \
  --recursive \
  --exclude "*" \
  --include "*.html" \
  --cache-control "public, max-age=0, must-revalidate" \
  --metadata-directive REPLACE
```

### Option B: Build Script

```bash
./deploy.sh YOUR-BUCKET-NAME
```

## S3 Bucket Configuration

### Static Website Hosting

1. Enable static website hosting on your S3 bucket
2. Set `index.html` as the index document
3. Set `404.html` (or `index.html`) as the error document

### Bucket Policy

Allow public read access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
    }
  ]
}
```

### CORS Configuration (if API is on different domain)

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedOrigins": ["*"],
    "ExposeHeaders": []
  }
]
```

## CloudFront Setup (Recommended)

For HTTPS and better performance, set up a CloudFront distribution:

1. Create a new CloudFront distribution
2. Origin: Your S3 bucket (use the website endpoint, not the REST API endpoint)
3. Viewer Protocol Policy: Redirect HTTP to HTTPS
4. Cache Policy: CachingOptimized
5. Origin Request Policy: CORS-S3Origin
6. Custom error pages: 404 → /index.html (for client-side routing)

## Backend API Requirements

The frontend expects a backend API at `NEXT_PUBLIC_API_URL` with these endpoints:

- `GET /health` - Health check
- `GET /meta` - API metadata
- `GET /sessions` - List sessions
- `POST /sessions` - Create session
- `GET /sessions/:id` - Get session
- `GET /sessions/:id/state` - Get session state
- `GET /sessions/:id/messages` - Get messages
- `POST /sessions/:id/messages` - Send message
- `GET /plans/:id` - Get plan
- `POST /plans/:id/approve` - Approve plan
- `POST /plans/:id/execute` - Execute plan
- `POST /plans/:id/clarify` - Clarify plan
- `GET /result-sets/:id` - Get result set
- `GET /documents/:id` - Get document metadata
- `GET /documents/:id/pdf` - Get PDF file
- `GET /evidence` - Get evidence context

### CORS

Ensure your backend allows CORS from your S3/CloudFront domain:

```python
# FastAPI example
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://friday.example.com",
        "https://d1234567890.cloudfront.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development

### Local Development

```bash
# Start development server with hot reload
npm run dev
```

The dev server runs at `http://localhost:3000`.

### API Proxy (Development)

For local development, you can run the backend on `localhost:8000` and set:

```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Troubleshooting

### API Connection Errors

1. Verify `NEXT_PUBLIC_API_URL` is set correctly before building
2. Check CORS configuration on the backend
3. Verify the backend is running and accessible

### 404 Errors on Page Refresh

S3 static hosting doesn't support client-side routing by default. Options:

1. Use CloudFront with custom error page pointing to `index.html`
2. Configure S3 error document to `index.html`

### PDF Viewer Not Loading

The PDF viewer uses an iframe. Ensure:

1. PDFs are served with correct CORS headers
2. `X-Frame-Options` allows embedding
