#!/bin/bash
# Consecutive 10-episode blocks for all 6 behavior profiles to prevent PyTorch OOM

# Array of experiments: ID, BEHAVIOR1, BEHAVIOR2
experiments=(
    "iql_selfish_100 selfish selfish"
    "iql_altruistic_100 altruistic altruistic"
    "iql_collaborative_100 collaborative collaborative"
    "iql_competitive_100 competitive competitive"
    "iql_malicious_100 malicious malicious"
    "iql_social_100 social social"
)

for exp in "${experiments[@]}"; do
    read -r exp_id b1 b2 <<< "$exp"
    echo "=========================================================="
    echo "Starting Experiment: $exp_id ($b1 / $b2)"
    echo "=========================================================="
    
    # 10 blocks of 10 episodes
    for start_ep in 1 11 21 31 41 51 61 71 81 91; do
        echo "--> Running Block: Episodes $start_ep to $((start_ep + 9))"
        
        source .venv/bin/activate && python scripts/iql_torchrl.py \
            --id "$exp_id" \
            --net Traveler_Info_7subreg_demand.mat \
            --episodes 10 \
            --start-episode "$start_ep" \
            --behavior1 "$b1" \
            --behavior2 "$b2" \
            --checkpoint-dir "results/$exp_id/checkpoints" \
            >> "results/${exp_id}.log" 2>&1
            
        echo "--> Block $start_ep completed and memory cleared."
    done
done

echo "All 6 behavior experiments completed in chunks!"
