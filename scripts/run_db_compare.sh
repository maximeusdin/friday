#!/bin/bash
# Run database comparison between local and AWS RDS

set -e

# Source AWS credentials
source ./friday_env.sh

# Run comparison
python scripts/compare_db_data.py \
  --local "postgresql://neh:neh@localhost:5432/neh" \
  --remote "$DATABASE_URL" \
  --verbose
