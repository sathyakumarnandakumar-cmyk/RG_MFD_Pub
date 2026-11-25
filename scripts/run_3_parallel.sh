#!/bin/bash
# 3 Parallel experiments, each chunked into 10-episode blocks

run_experiment() {
    exp_id=$1
    b1=$2
    b2=$3
    
    echo "Started $exp_id in background..."
    
    for start_ep in 1 11 21 31 41 51 61 71 81 91; do
        source .venv/bin/activate && python scripts/iql_torchrl.py \
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

# Run the first 3 behaviors from the table in parallel
run_experiment "iql_altruistic_100" "altruistic" "altruistic" &
run_experiment "iql_collaborative_100" "collaborative" "collaborative" &
run_experiment "iql_competitive_100" "competitive" "competitive" &

# Wait for all background processes to complete
wait
echo "All 3 parallel experiments successfully completed!"
