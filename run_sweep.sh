#!/bin/bash
set -e

PROJECT_NAME="privacy_preserving_federated_learning"

activate_venv () {
  SWEEP_NAME="$1"

  if [[ "$SWEEP_NAME" == *_dp || "$SWEEP_NAME" == *_centralized* ]]; then
    echo "[ENV] Using DP environment (venv_dp) for sweep: $SWEEP_NAME"
    source venv_dp/bin/activate
  else
    echo "[ENV] Using standard environment (venv) for sweep: $SWEEP_NAME"
    source venv/bin/activate
  fi

  python -c "import sys; print('[ENV] Python:', sys.executable)"
}

run_sweep () {
  SWEEP_NAME="$1"

  echo "=================================================="
  echo "[INFO] Creating sweep for: $SWEEP_NAME"
  echo "=================================================="

  # Activate correct environment
  deactivate >/dev/null 2>&1 || true
  activate_venv "$SWEEP_NAME"

  # Create sweep
  OUTPUT=$(wandb sweep \
    --project "$PROJECT_NAME" \
    --name "$SWEEP_NAME" \
    "config/$SWEEP_NAME.yaml" 2>&1)

  echo "[DEBUG] wandb sweep output:"
  echo "$OUTPUT"

  # Extract sweep ID
  SWEEP_ID=$(echo "$OUTPUT" | grep -oE '[^ ]+/[^ ]+/[a-zA-Z0-9]{8}' | tail -n1)

  if [ -z "$SWEEP_ID" ]; then
    echo "[ERROR] Failed to extract sweep ID."
    exit 1
  fi

  echo "[INFO] Launching agent for sweep ID: $SWEEP_ID"
  wandb agent "$SWEEP_ID"
}

# ==================================================
# BODY SIGNAL OF SMOKING
# ==================================================
run_sweep "smoking_centralized"        # No FL, No DP
run_sweep "smoking_dp"                  # No FL, DP
# run_sweep "smoking_fl"
# run_sweep "smoking_fl_dp_central"
#run_sweep "smoking_fl_dp_local"

# ==================================================
# CIFAR-10
# ==================================================
#run_sweep "cifar_centralized"
# run_sweep "cifar_dp"
# run_sweep "cifar_fl"
# run_sweep "cifar_fl_dp_local"
# run_sweep "cifar_fl_dp_central"

# ==================================================
# NETWORK MONITORING
# ==================================================
# run_sweep "network_centralized"
# run_sweep "network_dp"
# run_sweep "network_fl"
# run_sweep "network_fl_dp_local"
# run_sweep "network_fl_dp_central"
