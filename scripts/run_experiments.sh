#!/bin/bash

# Ensure we use the virtual environment
source .venv/bin/activate

# Base command
BASE_CMD="python scripts/iql_torchrl.py --net Traveler_Info_7subreg_demand.mat --episodes 1"

echo "========================================="
echo "Starting MARL Symmetric Behavior Runs"
echo "========================================="

# 1. Altruistic: Minimize delay for everyone
echo "1. Running Altruistic..."
$BASE_CMD --id iql_run_altruistic --behavior1 altruistic --behavior2 altruistic

# 2. Collaborative: Minimize delay for oneself and one’s own group
echo "2. Running Collaborative..."
$BASE_CMD --id iql_run_collaborative --behavior1 collaborative --behavior2 collaborative

# 3. Competitive: Minimize self-delay and maximize for the other group
echo "3. Running Competitive..."
$BASE_CMD --id iql_run_competitive --behavior1 competitive --behavior2 competitive

# 4. Malicious: Maximize delay for the other group
echo "4. Running Malicious..."
$BASE_CMD --id iql_run_malicious --behavior1 malicious --behavior2 malicious

# 5. Selfish: Minimize delay for oneself
echo "5. Running Selfish..."
$BASE_CMD --id iql_run_selfish --behavior1 selfish --behavior2 selfish

# 6. Social: Minimize delay for oneself and everyone
echo "6. Running Social..."
$BASE_CMD --id iql_run_social --behavior1 social --behavior2 social

echo "========================================="
echo "All 6 experiments completed!"
echo "========================================="
