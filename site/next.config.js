/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export for S3 hosting
  output: 'export',
  
  // Disable image optimization (not supported in static export)
  images: {
    unoptimized: true,
  },

  // Trailing slashes for S3 compatibility
  trailingSlash: true,

  // Webpack config for PDF.js worker
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },

  // Environment variables to bake into the build
  env: {
    // Set this to your deployed backend API URL
    // e.g., https://api.friday.example.com or API Gateway URL
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || '',
  },
};

module.exports = nextConfig;
