#!/bin/bash
# Setup script for NER dependencies

echo "Installing SpaCy..."
pip install spacy>=3.7.0

echo "Downloading SpaCy models..."
python -m spacy download en_core_web_sm

echo "Optional: Download transformer model (slower but more accurate):"
echo "  python -m spacy download en_core_web_trf"

echo "Setup complete!"
