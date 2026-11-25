#!/bin/bash
# MAPPO: 3 parallel experiments, each chunked into 10-episode blocks

run_experiment() {
    exp_id=$1
    b1=$2
    b2=$3
    
    echo "Started MAPPO $exp_id..."
    
    for start_ep in 1 11 21 31 41 51 61 71 81 91; do
        source .venv/bin/activate && python scripts/mappo_central.py \
            --id "$exp_id" \
            --net Traveler_Info_7subreg_demand.mat \
            --episodes 10 \
            --start-episode "$start_ep" \
            --behavior1 "$b1" \
            --behavior2 "$b2" \
            --checkpoint-dir "results/$exp_id/checkpoints" \
            >> "results/${exp_id}.log" 2>&1
    done
    
    echo "$exp_id Finished!"
}

# First batch: 3 parallel
run_experiment "mappo_altruistic_100" "altruistic" "altruistic" &
run_experiment "mappo_collaborative_100" "collaborative" "collaborative" &
run_experiment "mappo_competitive_100" "competitive" "competitive" &
wait
echo "Batch 1 done!"

# Second batch: 3 parallel
run_experiment "mappo_malicious_100" "malicious" "malicious" &
run_experiment "mappo_selfish_100" "selfish" "selfish" &
run_experiment "mappo_social_100" "social" "social" &
wait
echo "All 6 MAPPO experiments completed!"
