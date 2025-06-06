#!/bin/bash

# This script runs experiments for PointNet++ on ModelNet datasets with optimized hyperparameters

# Default parameters
BATCH_SIZE=32
NUM_EPOCHS=200
SEED=42
NUM_POINTS=1024
SAVE_DIR="models/modelnet/pointnet++"
DATASET_PATH="data"
LEARNING_RATE=0.001
SA_RATIO_1=0.5    # First set abstraction sampling ratio
SA_RATIO_2=0.25   # Second set abstraction sampling ratio
SA_RADIUS_1=0.2   # First set abstraction radius
SA_RADIUS_2=0.4   # Second set abstraction radius
FEATURE_DIM_1=128 # Feature dimension after first set abstraction
FEATURE_DIM_2=256 # Feature dimension after second set abstraction
FEATURE_DIM_3=1024 # Feature dimension after global set abstraction
DROPOUT=0.3       # Dropout rate

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
    --sa_ratio_1)
      SA_RATIO_1="$2"
      shift 2
      ;;
    --sa_ratio_2)
      SA_RATIO_2="$2"
      shift 2
      ;;
    --sa_radius_1)
      SA_RADIUS_1="$2"
      shift 2
      ;;
    --sa_radius_2)
      SA_RADIUS_2="$2"
      shift 2
      ;;
    --feature_dim_1)
      FEATURE_DIM_1="$2"
      shift 2
      ;;
    --feature_dim_2)
      FEATURE_DIM_2="$2"
      shift 2
      ;;
    --feature_dim_3)
      FEATURE_DIM_3="$2"
      shift 2
      ;;
    --dropout)
      DROPOUT="$2"
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
NAME="PointNet++_modelnet${MODELNET_VERSION}_optimized"
echo "Running PointNet++ experiment on ModelNet${MODELNET_VERSION}"
echo "====================================="

python modelnet_run.py \
    --model PointNet2 \
    --modelnet_version $MODELNET_VERSION \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --num_points $NUM_POINTS \
    --seed $SEED \
    --save_dir $SAVE_DIR \
    --dataset_path $DATASET_PATH \
    --name $NAME \
    --lr $LEARNING_RATE \
    --set_abstraction_ratio_1 $SA_RATIO_1 \
    --set_abstraction_ratio_2 $SA_RATIO_2 \
    --set_abstraction_radius_1 $SA_RADIUS_1 \
    --set_abstraction_radius_2 $SA_RADIUS_2 \
    --feature_dim_1 $FEATURE_DIM_1 \
    --feature_dim_2 $FEATURE_DIM_2 \
    --feature_dim_3 $FEATURE_DIM_3 \
    --dropout $DROPOUT
    
echo "Experiment completed: $NAME"
echo "=====================================" 