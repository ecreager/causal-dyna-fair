#!/bin/bash
#
# NOTE: This script requires the enivonrment variable EXP_DIR

num_samps=10000

rm -f /tmp/commands.txt

################################################################################
# BASELINE COMMANDS
################################################################################
# generate commands
policy_intervention=False
repayment_intervention=False
tag="baseline"
#for num_steps in 10 20 30 40
for num_steps in $(seq 1 9)
do
  exp_name=multi_step_repayment_intervention.$tag.N$num_samps.S$num_steps
  results_dir=$EXP_DIR/$exp_name
  echo python -W ignore multi_step_repayment_intervention.py \
    --gin_file=\"config/multi_step_repayment_intervention.gin\" \
    --gin_param=\"query_parameters.num_samps="$num_samps"\" \
    --gin_param=\"query_parameters.num_steps="$num_steps"\" \
    --gin_param=\"query_parameters.results_dir=\'$results_dir\'\" \
    --gin_param=\"get_intervened_simulation.policy_intervention="$policy_intervention"\" \
    --gin_param=\"get_intervened_simulation.repayment_intervention="$repayment_intervention"\" >> /tmp/commands.txt
done

################################################################################
# POLICY INTERVENTION COMMANDS
################################################################################
# generate commands
policy_intervention=True
repayment_intervention=False
tag="policy-intervention"
#for num_steps in 10 20 30 40
for num_steps in $(seq 1 9)
do
  exp_name=multi_step_repayment_intervention.$tag.N$num_samps.S$num_steps
  results_dir=$EXP_DIR/$exp_name
  echo python -W ignore multi_step_repayment_intervention.py \
    --gin_file=\"config/multi_step_repayment_intervention.gin\" \
    --gin_param=\"query_parameters.num_samps="$num_samps"\" \
    --gin_param=\"query_parameters.num_steps="$num_steps"\" \
    --gin_param=\"query_parameters.results_dir=\'$results_dir\'\" \
    --gin_param=\"get_intervened_simulation.policy_intervention="$policy_intervention"\" \
    --gin_param=\"get_intervened_simulation.repayment_intervention="$repayment_intervention"\" >> /tmp/commands.txt
done

################################################################################
# REPAYMENT INTERVENTION COMMANDS
################################################################################
# generate commands
policy_intervention=True
repayment_intervention=True
tag="repayment-intervention"
#for num_steps in 10 20 30 40
for num_steps in $(seq 1 9)
do
  exp_name=multi_step_repayment_intervention.$tag.N$num_samps.S$num_steps
  results_dir=$EXP_DIR/$exp_name
  echo python -W ignore multi_step_repayment_intervention.py \
    --gin_file=\"config/multi_step_repayment_intervention.gin\" \
    --gin_param=\"query_parameters.num_samps="$num_samps"\" \
    --gin_param=\"query_parameters.num_steps="$num_steps"\" \
    --gin_param=\"query_parameters.results_dir=\'$results_dir\'\" \
    --gin_param=\"get_intervened_simulation.policy_intervention="$policy_intervention"\" \
    --gin_param=\"get_intervened_simulation.repayment_intervention="$repayment_intervention"\" >> /tmp/commands.txt
done

################################################################################
# EXECUTE ALL COMMANDS
################################################################################
source /tmp/commands.txt
