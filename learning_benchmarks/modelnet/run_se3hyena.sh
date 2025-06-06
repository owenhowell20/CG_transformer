#!/bin/bash

# This script runs experiments for SE3Hyena on ModelNet datasets with optimized hyperparameters

# Default parameters
BATCH_SIZE=32 # Smaller batch size for better training
NUM_EPOCHS=200
SEED=42
NUM_POINTS=1024
SAVE_DIR="models/modelnet/se3hyena"
DATASET_PATH="data"
LEARNING_RATE=0.001  
POS_ENC_DIM=128  # Increased positional encoding dimension
INPUT_DIM_1=256  # Adjusted model dimensions
INPUT_DIM_2=128
INPUT_DIM_3=64

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
NAME="SE3Hyena_modelnet${MODELNET_VERSION}_optimized"
echo "Running SE3Hyena experiment on ModelNet${MODELNET_VERSION}"
echo "====================================="

python modelnet_run.py \
    --model SE3Hyena \
    --modelnet_version $MODELNET_VERSION \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --num_points $NUM_POINTS \
    --seed $SEED \
    --save_dir $SAVE_DIR \
    --dataset_path $DATASET_PATH \
    --name $NAME \
    --lr $LEARNING_RATE \
    --positional_encoding_dimension $POS_ENC_DIM \
    --input_dimension_1 $INPUT_DIM_1 \
    --input_dimension_2 $INPUT_DIM_2 \
    --input_dimension_3 $INPUT_DIM_3
    
echo "Experiment completed: $NAME"
echo "=====================================" 