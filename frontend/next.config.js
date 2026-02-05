/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for containerized deployments
  output: 'standalone',

  // Environment variables to bake into the build
  env: {
    // Production API URL - must be HTTPS
    // Can be overridden at build time via environment variable
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://api.fridayarchive.org/api',
  },
  
  // Configure API proxy for development (only used when API_BASE is relative '/api')
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.API_URL || 'http://localhost:8000/api/:path*',
      },
    ];
  },

  // Webpack config for PDF.js worker
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },

  // Extend proxy timeout for long-running V6 queries (10 minutes)
  experimental: {
    proxyTimeout: 600000, // 10 minutes in milliseconds
  },

  // Increase server timeout for API routes
  serverRuntimeConfig: {
    apiTimeout: 600000,
  },
};

module.exports = nextConfig;
