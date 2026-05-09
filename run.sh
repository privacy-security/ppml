#!/bin/bash
normalize_bool() {
    local val=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if [[ "$val" == "true" ]]; then
        echo true
    else
        echo false
    fi
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/venv/bin/:${SCRIPT_DIR}"
echo "[RUNNER] PYTHONPATH set to: $PYTHONPATH"
echo "[RUNNER] Working directory: $(pwd)"

# Needed for MacOS to work run the `pkill` properly
export LC_CTYPE=C
export LANG=C

# Kill any currently running client.py processes
pkill -f 'flower-client-app'
# Kill any currently running flower-superlink processes
pkill -f 'flower-superlink'

sleep 10

# Parse arguments
for arg in "$@"
do
    case $arg in
        --aggregation_type=*) AGGREGATION_TYPE="${arg#*=}" ;;
        --dp=*) DP="${arg#*=}" ;;
        --partition_type=*) PARTITION_TYPE="${arg#*=}" ;;
        --l2_norm_clip=*) L2_NORM_CLIP="${arg#*=}" ;;
        --noise_multiplier=*) NOISE_MULTIPLIER="${arg#*=}" ;;
        --learning_rate=*) LEARNING_RATE="${arg#*=}" ;;
        --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
        --epochs=*) EPOCHS="${arg#*=}" ;;
        --local=*) LOCAL_DP="${arg#*=}" ;;
        --clipping=*) CLIPPING="${arg#*=}" ;;
        --epsilon=*) EPSILON="${arg#*=}" ;;
        --delta=*) DELTA="${arg#*=}" ;;
        --no_fl=*) NO_FL="${arg#*=}" ;;
        --n_clients=*) N_CLIENTS="${arg#*=}" ;;
        --dataset=*) DATASET="${arg#*=}" ;;
        *) ;;
    esac
done

# Dataset
export DATASET
DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

# Differential Privacy
export DP
export LOCAL_DP
export CLIPPING
export SENSITIVITY
export EPSILON
export DELTA
export L2_NORM_CLIP
export NOISE_MULTIPLIER
export AGGREGATION_TYPE
export PARTITION_TYPE
export LEARNING_RATE
export BATCH_SIZE
export EPOCHS
# Federated Learning
export NO_FL
export N_CLIENTS

# Normalize
DP=$(normalize_bool "$DP")
LOCAL_DP=$(normalize_bool "$LOCAL_DP")
NO_FL=$(normalize_bool "$NO_FL")

echo "[RUNNER] Using DP: $DP"
if [ "$DP" = true ]; then
    if [ "$LOCAL_DP" = true ]; then
        echo "[RUNNER] Using L2_NORM_CLIP: $L2_NORM_CLIP"
        echo "[RUNNER] Using SENSITIVITY: $SENSITIVITY"
        echo "[RUNNER] Using EPSILON: $EPSILON"
        echo "[RUNNER] Using DELTA: $DELTA"
    fi
    if [ "$LOCAL_DP" = false ]; then
        echo "[RUNNER] Using CLIPPING: $CLIPPING"
        echo "[RUNNER] Using L2_NORM_CLIP: $L2_NORM_CLIP"
        echo "[RUNNER] Using NOISE_MULTIPLIER: $NOISE_MULTIPLIER"
    fi
fi
echo "[RUNNER] Learning rate: $LEARNING_RATE"
echo "[RUNNER] Batch size: $BATCH_SIZE"
echo "[RUNNER] Epochs: $EPOCHS"
echo "[RUNNER] Using FL: $([ "$NO_FL" = true ] && echo false || echo true)"

if [ "$NO_FL" = true ]; then
    echo "[RUNNER] Running centralized (non-FL) training..."
    python "$SCRIPT_DIR/train_centralized.py"
    exit 0
fi

# Start the flower server
#echo "[RUNNER] Starting flower server in background..."
flower-superlink --insecure > /dev/null 2>&1 &
sleep 10

# Partition Types
echo "[RUNNER] Partition type from environment: $PARTITION_TYPE" # (1)-(2)-(3);
# export PARTITION_TYPE="iid" # (123)-(123)-(123);
# export PARTITION_TYPE="vertical" # per-feature (only 2 clients)
echo "[RUNNER] Using $PARTITION_TYPE Partition Type"

# Aggregation Type
echo "[RUNNER] Using $AGGREGATION_TYPE Aggregation Type"

# Start resource measurement in the background
# source ./.measure_resource.sh
# measure_resources &
# bg_pid=$!
# echo "[RUNNER] Background Process PID: $bg_pid"

# Start N client processes
echo "[RUNNER] Starting $N_CLIENTS ClientApps"
for i in $(seq 1 $N_CLIENTS)
do
    export CID=$((i-1)) &&
    echo "[RUNNER] Starting ClientApp #$i" &&
    flower-client-app --insecure client_app:app &
    sleep 0.1
done

echo "[RUNNER] Starting ServerApp..."
# Enable maximum verbosity and logging
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=0
flower-server-app --insecure server_app:app --verbose 2>&1 | tee server_log.txt

echo "[RUNNER] Clearing background processes..."
# Kill any currently running client.py processes
pkill -f 'flower-client-app' 2>/dev/null
# Kill any currently running flower-superlink processes
pkill -f 'flower-superlink' 2>/dev/null
# Only kill bg_pid if it exists
if [ ! -z "$bg_pid" ]; then
    pkill -P $bg_pid 2>/dev/null
fi
pkill -P $$ 2>/dev/null
