#!/bin/bash

# This script runs experiments for DGCNN on ModelNet datasets with optimized hyperparameters

# Default parameters
BATCH_SIZE=32  # DGCNN works well with larger batches
NUM_EPOCHS=200
SEED=42
NUM_POINTS=1024
SAVE_DIR="models/modelnet/dgcnn"
DATASET_PATH="data"
LEARNING_RATE=0.001
K=20  # k-nearest neighbors parameter

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --modelnet)
      MODELNET_VERSION="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Default to ModelNet40 if not specified
if [ -z "$MODELNET_VERSION" ]; then
  MODELNET_VERSION="40"
fi

# Run experiment
NAME="DGCNN_modelnet${MODELNET_VERSION}_optimized"
echo "Running DGCNN experiment on ModelNet${MODELNET_VERSION}"
echo "====================================="

python modelnet_run.py \
    --model DGCNN \
    --modelnet_version $MODELNET_VERSION \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --num_points $NUM_POINTS \
    --seed $SEED \
    --save_dir $SAVE_DIR \
    --dataset_path $DATASET_PATH \
    --name $NAME \
    --lr $LEARNING_RATE \
    --k $K
    
echo "Experiment completed: $NAME"
echo "=====================================" 