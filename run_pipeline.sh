#!/bin/bash
echo "Running tablab pipeline"

# Activate your venv
source tablab/bin/activate

# Download the dataset
python regression.py

# Preprocess the data
# python classification.py

# Run the main script
# python -m scripts.run_benchmark --config configs/config-reg.yaml

# chmod +x run_pipeline.sh