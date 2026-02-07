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
    // Production API URL - must be HTTPS
    // Can be overridden at build time via environment variable
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://api.fridayarchive.org/api',
  },

  // Webpack config for PDF.js worker
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
};

module.exports = nextConfig;
