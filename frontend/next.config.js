/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for containerized deployments
  output: 'standalone',
  
  // Configure API proxy for development
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
};

module.exports = nextConfig;
