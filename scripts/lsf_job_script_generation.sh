#!/bin/bash

# LSF Job Submission Script with Dynamic Parameters
# Usage: ./scripts/lsf_job_script_generation.sh -n <cores> -g <gpus> [other options]

# Default values
CORES=8
GPUS=1
EXPERIMENT="segmentation_r2s100k"
MODEL="dino_vits8-linear_probing"
EPOCHS=500
BATCH_SIZE=128
NUM_WORKERS=8
QUEUE="gpu"
JOB_NAME="mt-vfmseg"
FREEZE=true  
USE_AMP=false
GRAD_CLIP=""
LR="0.001"
RESUME=""
# Script paths
PROJECT_DIR="/home/phd_li/git_repo/MT-DinoSeg"
LOG_DIR="$PROJECT_DIR/logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--cores)
            CORES="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -q|--queue)
            QUEUE="$2"
            shift 2
            ;;
        -j|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --no-freeze)
            FREEZE=false
            shift
            ;;
        --use-amp|--enable-amp)
            USE_AMP=true
            shift
            ;;
        --grad-clip)
            GRAD_CLIP="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -n, --cores        Number of CPU cores (default: $CORES)"
            echo "  -g, --gpus         Number of GPUs (default: $GPUS)"
            echo "  -e, --experiment   Experiment name (default: $EXPERIMENT)"
            echo "  -m, --model        Model name (default: $MODEL)"
            echo "  --epochs           Number of epochs (default: $EPOCHS)"
            echo "  --batch-size       Batch size (default: $BATCH_SIZE)"
            echo "  --num-workers      Number of workers (default: $NUM_WORKERS)"
            echo "  -q, --queue        LSF queue (default: $QUEUE)"
            echo "  -j, --job-name     Job name (default: $JOB_NAME)"
            echo ""
            echo "Training Options:"
            echo "  --no-freeze        Don't freeze encoder weights (default)"
            echo "  --use-amp          Enable mixed precision training"
            echo "  --grad-clip VALUE  Gradient clipping value (e.g., 1.0)"
            echo "  --lr               Learning Rate (default: $LR)"
            echo "  --resume PATH      Path to the checkpoint (default: $RESUME)"
            echo ""
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ $GPUS -lt 1 ]; then
    echo "Error: Number of GPUs must be at least 1"
    exit 1
fi

if [ $CORES -lt 1 ]; then
    echo "Error: Number of cores must be at least 1"
    exit 1
fi

# Calculate optimal cores per GPU (typically 8-16 cores per GPU)
CORES_PER_GPU=$((CORES / GPUS))
if [ $CORES_PER_GPU -lt 4 ]; then
    echo "Warning: Only $CORES_PER_GPU cores per GPU. Consider increasing total cores."
fi

if [ "$USE_AMP" = true ]; then
    echo "Mixed Precision enabled - you can use larger batch sizes!"
    if [ $BATCH_SIZE -lt 32 ]; then
        echo "   💡 Tip: Consider increasing batch size (current: $BATCH_SIZE, suggested: 32+)"
    fi
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate job script
JOB_SCRIPT="$PROJECT_DIR/job_${JOB_NAME}_${GPUS}gpu.sh"

cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#BSUB -J ${JOB_NAME}_${GPUS}gpu
#BSUB -o ${LOG_DIR}/${EXPERIMENT}_${MODEL}_%J.log
#BSUB -e ${LOG_DIR}/${EXPERIMENT}_${MODEL}_%J.log
#BSUB -n ${CORES}
#BSUB -q ${QUEUE}
#BSUB -gpu "num=${GPUS}:j_exclusive=yes"
#BSUB -R "select[type==X64LIN]"

# Configuration
MASTER_PORT=\$((29500 + LSB_JOBID % 1000))
NPROC_PER_NODE=${GPUS}

echo "=== MT-DinoSeg Training Job ==="
echo "Date: \$(date)"
echo "Host: \$(hostname)"
echo "Job ID: \$LSB_JOBID"
echo "Master Port: \$MASTER_PORT"
echo "CPUs: ${CORES}"
echo "GPUs: ${GPUS}"
echo "Experiment: ${EXPERIMENT}"
echo "Model: ${MODEL}"
echo "Mixed Precision: ${USE_AMP}"
echo "Freeze Encoder: ${FREEZE}"
$(if [ -n "$GRAD_CLIP" ]; then echo "echo \"Gradient Clipping: ${GRAD_CLIP}\""; fi)
echo ""

echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

echo "Memory and CPU info:"
echo "CPUs: \$(nproc)"
echo "Memory: \$(free -h | grep Mem | awk '{print \$2}')"
echo ""

# Change to project directory
cd ${PROJECT_DIR}

# Set up environment
export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --list-gpus | wc -l | awk '{for(i=0;i<\$1;i++) printf i (i<\$1-1 ? "," : "")}')

TRAIN_CMD="train.py \\\\
    --experiment ${EXPERIMENT} \\\\
    --model_name ${MODEL} \\\\
    --epochs ${EPOCHS} \\\\
    --batch_size ${BATCH_SIZE} \\\\
    --learning_rate ${LR} \\\\
    --num_workers ${NUM_WORKERS}"

# Add optional flags
$(if [ "$FREEZE" = false ]; then echo "TRAIN_CMD=\"\$TRAIN_CMD --train_backbone\""; fi)
$(if [ "$USE_AMP" = true ]; then echo "TRAIN_CMD=\"\$TRAIN_CMD --use_amp\""; fi)
$(if [ -n "$GRAD_CLIP" ]; then echo "TRAIN_CMD=\"\$TRAIN_CMD --grad_clip ${GRAD_CLIP}\""; fi)
$(if [ "$RESUME" != "" ]; then echo "TRAIN_CMD=\"\$TRAIN_CMD --resume_from ${RESUME}\""; fi)

# Run training
if [ ${GPUS} -eq 1 ]; then
    echo "Running single GPU training..."
    echo "Command: python \$TRAIN_CMD"
    echo ""
    eval "python \$TRAIN_CMD"
else
    echo "Running multi-GPU training with \$NPROC_PER_NODE GPUs..."
    echo "Command: torchrun --standalone --nproc-per-node=\$NPROC_PER_NODE \$TRAIN_CMD"
    echo ""
    eval "torchrun --standalone --nproc-per-node=\$NPROC_PER_NODE \$TRAIN_CMD"
fi


echo ""
echo "Training completed at \$(date)"
EOF

# Make the job script executable
chmod +x "$JOB_SCRIPT"

# Display information
echo "Generated job script: $JOB_SCRIPT"
echo "Job configuration:"
echo "  Job Name: ${JOB_NAME}_${GPUS}gpu"
echo "  Cores: $CORES"
echo "  GPUs: $GPUS"
echo "  Queue: $QUEUE"
echo "  Experiment: $EXPERIMENT"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Mixed Precision: $USE_AMP"
echo "  Freeze Encoder: $FREEZE"
if [ -n "$GRAD_CLIP" ]; then
    echo "  Gradient Clipping: $GRAD_CLIP"
fi
echo ""

bsub < "$JOB_SCRIPT"

# Submit the job
# read -p "Submit job to LSF? (y/n) [y]: " submit_job
# if [[ ! $submit_job =~ ^[Nn]$ ]]; then
#     echo "Submitting job..."
#     bsub < "$JOB_SCRIPT"
    
#     echo ""
#     echo "Job submitted! Monitor with:"
#     echo "  bjobs          # List your jobs"
#     echo "  bhist -l       # Job history"
#     echo "  bpeek <jobid>  # Peek at job output"
# else
#     echo "Job script created but not submitted."
#     echo "Submit manually with: bsub < $JOB_SCRIPT"
# fi