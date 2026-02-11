/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export for S3 hosting (use 'standalone' for containerized deployments)
  output: 'export',

  // Disable image optimization (not supported in static export)
  images: {
    unoptimized: true,
  },

  // Trailing slashes for S3 compatibility
  trailingSlash: true,

  // Environment variables to bake into the build
  env: {
    // API URL: defaults to localhost in dev, production in prod
    // Override via NEXT_PUBLIC_API_URL (e.g. .env.local for local backend)
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL
      || (process.env.NODE_ENV === 'development' ? 'http://localhost:8000/api' : 'https://api.fridayarchive.org/api'),
  },

  // Webpack config for PDF.js worker
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
};

module.exports = nextConfig;
