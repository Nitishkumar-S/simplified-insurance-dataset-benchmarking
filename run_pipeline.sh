#!/bin/bash
echo "Running tablab pipeline"

# Activate your venv
source tablab/bin/activate

# install requirements.txt if not worked from job.slurm 'a fallback'
pip install -r requirements.txt

# Download the dataset
# python regression.py --dataset "$1" --target "$2"

# Preprocess the data
python classification.py --dataset "$1"

# Run the main script
# python -m scripts.run_benchmark --config configs/config-reg.yaml

# chmod +x run_pipeline.sh
