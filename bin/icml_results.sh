#!/bin/bash
# REPRODUCES RESULTS FIGURES FROM THE ICML PAPER
#
# NOTE: Run this from the root repo like ./bin/icml_results.sh

set -e  # exit if any errors arise

TAG=$RANDOM$RANDOM  # random tag
ROOTDIR=/tmp/causal-dyna-fair/$TAG  # root directory for all results
mkdir -p $ROOTDIR
 
################################################################################
# FIGURES 5 AND 6
################################################################################
FIG5_FIG6_DIR=$ROOTDIR/figure5_figure6
mkdir -p $FIG5_FIG6_DIR
OBS_DATA_DIR=$ROOTDIR/obs-data
mkdir -p $OBS_DATA_DIR

# generate observational data
for seed in $(seq 0 12)
do
  echo Generating observational dataset $seed out of 12
  python -W ignore observational_data.py \
    --gin_file=./config/observational_data.gin \
    --gin_param=seed=$seed \
    --gin_param=results_dir="'$OBS_DATA_DIR'"
done

# perform off-policy analysis
python -W ignore off_policy_eval.py \
    --obs_data_dir $OBS_DATA_DIR \
    --results_dir $FIG5_FIG6_DIR
  
#################################################################################
# FIGURE 7
################################################################################
FIG7_DIR=$ROOTDIR/figure7
mkdir -p $FIG7_DIR
python -W ignore credit_score_intervention.py \
  --gin_file=config/credit_score_intervention.gin \
  --gin_param=num_samps=10000 \
  --gin_param=results_dir="'$FIG7_DIR'"
 
################################################################################
# FIGURE 9
################################################################################
FIG9_DIR=$ROOTDIR/figure9
mkdir -p $FIG9_DIR
EXP_DIR=$FIG9_DIR
source bin/multi_step_repayment_intervention.sh
python -W ignore plot_multi_step_repayment_intervention.py --expdir=$FIG9_DIR
