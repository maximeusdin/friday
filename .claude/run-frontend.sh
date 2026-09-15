#!/bin/bash
# Dev-server launcher for the Claude Code preview pane.
# Node/npm live only in the friday conda env on this Mac.
export PATH=/opt/anaconda3/envs/friday/bin:$PATH
# Own build dir, so a production build or deploy can't pull the rug out.
export NEXT_DIST_DIR=.next-dev
cd /Users/maxime/friday/frontend
exec npm run dev
