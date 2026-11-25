#!/bin/bash

echo "Starting Batch 1 of the MARL Hyperparameter Grid (marl_1 to marl_4)..."

# marl_1: lr=1e-3, tau=0.005, buffer=10000
tmux new -s marl_1 -d 'source .venv/bin/activate && python scripts/iql_torchrl.py --id marl_1 --net Traveler_Info_7subreg_demand.mat --episodes 100 --behavior1 altruistic --behavior2 altruistic --lr 1e-3 --tau 0.005 --buffer-size 10000 --annealing-steps 1500000'
echo "Started marl_1"

# marl_2: lr=1e-3, tau=0.005, buffer=50000
tmux new -s marl_2 -d 'source .venv/bin/activate && python scripts/iql_torchrl.py --id marl_2 --net Traveler_Info_7subreg_demand.mat --episodes 100 --behavior1 altruistic --behavior2 altruistic --lr 1e-3 --tau 0.005 --buffer-size 50000 --annealing-steps 1500000'
echo "Started marl_2"

# marl_3: lr=1e-3, tau=0.001, buffer=10000
tmux new -s marl_3 -d 'source .venv/bin/activate && python scripts/iql_torchrl.py --id marl_3 --net Traveler_Info_7subreg_demand.mat --episodes 100 --behavior1 altruistic --behavior2 altruistic --lr 1e-3 --tau 0.001 --buffer-size 10000 --annealing-steps 1500000'
echo "Started marl_3"

# marl_4: lr=1e-3, tau=0.001, buffer=50000
tmux new -s marl_4 -d 'source .venv/bin/activate && python scripts/iql_torchrl.py --id marl_4 --net Traveler_Info_7subreg_demand.mat --episodes 100 --behavior1 altruistic --behavior2 altruistic --lr 1e-3 --tau 0.001 --buffer-size 50000 --annealing-steps 1500000'
echo "Started marl_4"

echo "Batch 1 is running!"
