#!/bin/bash

# Launching tune_low_lr in a detached tmux session
tmux new -s tune_low_lr -d 'source .venv/bin/activate && python scripts/iql_torchrl.py \
    --id "tune_low_lr" \
    --net Traveler_Info_7subreg_demand.mat \
    --episodes 100 \
    --behavior1 "altruistic" \
    --behavior2 "altruistic" \
    --lr 1e-4 \
    --annealing-steps 1500000 \
    --update-interval 4'
echo "Started tmux session: tune_low_lr"

# Launching tune_lazy_train in a detached tmux session
tmux new -s tune_lazy_train -d 'source .venv/bin/activate && python scripts/iql_torchrl.py \
    --id "tune_lazy_train" \
    --net Traveler_Info_7subreg_demand.mat \
    --episodes 100 \
    --behavior1 "altruistic" \
    --behavior2 "altruistic" \
    --lr 1e-4 \
    --annealing-steps 1500000 \
    --update-interval 16'
echo "Started tmux session: tune_lazy_train"

echo "Both experiments are now running in the background."
echo "Use 'tmux ls' to see them, and 'tmux attach -t <session_name>' to view progress."
