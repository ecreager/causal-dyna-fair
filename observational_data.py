"""Generate observational data."""
import os
import pickle
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import pandas as pd

import whynot as wn
from whynot.simulators.delayed_impact.simulator import INV_CDFS
from whynot.simulators.delayed_impact.simulator import GROUP_SIZE_RATIO

from utils.data import get_data_args
from utils.policy import get_policy

SOURCE_POLICY_NAME = 'maxprof'
TARGET_POLICY_NAME = 'dempar'

Array = np.ndarray


def log(msg: str):
    """Log stuff."""
    print(msg)
    logging.info(msg)


@gin.configurable
def get_experiment(
        policy_name: str = None,
        repayment_utility: float = gin.REQUIRED,
        default_utility: float = gin.REQUIRED,
        repayment_score_change: float = gin.REQUIRED,
        default_score_change: float = gin.REQUIRED,
        num_steps: int = gin.REQUIRED,
) -> Tuple:
    """Produce an observational dataset from simulation given config params."""

    def sample_initial_states(rng):
        """ Sample initial states.

        Sample initial states used on FICO data. Each initial state
        corresponds to an agent.
        """
        group = int(rng.uniform() < GROUP_SIZE_RATIO[1])
        # Compute credit score via inverse CDF trick
        score = INV_CDFS[group](rng.uniform())
        return wn.delayed_impact.State(group=group, credit_score=score)

    data_args = get_data_args()
    (_, loan_repaid_probs, pis, group_size_ratio, scores_list, \
            _) = data_args
    utils = (default_utility, repayment_utility)
    impact = (default_score_change, repayment_score_change)
    f_T = get_policy(loan_repaid_probs, pis, group_size_ratio, utils, impact,
                     scores_list, policy_name=policy_name)
    threshold_g0 = f_T.threshold_group_0
    threshold_g1 = f_T.threshold_group_1
    log(
        'under policy %s, threshold_group0 = %.2f, threshold_group1 = %.2f' % (
            policy_name, threshold_g0, threshold_g1)
    )

    config = wn.delayed_impact.Config(
        start_time=0, end_time=num_steps,
        repayment_utility=repayment_utility,
        default_utility=default_utility,
        repayment_score_change=repayment_score_change,
        default_score_change=default_score_change,
        threshold_g0=threshold_g0, threshold_g1=threshold_g1
    )

    def extract_outcomes(run):
        """Extract outcomes.

        Outcome is score change Delta and the institutions profit after 1 step.
        """
        return [run.states[1].credit_score - run.states[0].credit_score,
                run.states[1].profits,
                run.states[1].loan_approved,
                run.states[1].repaid]

    # Construct the experiment
    description = "For generating observational data for off-policy analysis."
    LendingExperiment = wn.DynamicsExperiment(
        name="LendingExperiment",
        description=description,
        simulator=wn.delayed_impact,
        simulator_config=config,
        # Change the credit scoring mechanism on the first step.
        intervention=wn.delayed_impact.Intervention(time=0),  # no intervention
        state_sampler=sample_initial_states,
        propensity_scorer=1.0,  # all "treated" i.e. processed by bank policy
        outcome_extractor=extract_outcomes,
        # Only covariate is group membership, which is a confounder for this
        # experiment.
        covariate_builder=lambda run: [run.initial_state.group,
                                       run.initial_state.credit_score,
                                       ]
    )

    dynamics = (utils, impact)
    return LendingExperiment, config, data_args, dynamics, f_T


def get_dataframe(dataset: wn.framework.Dataset) -> pd.DataFrame:
    """Produce pandas two DataFrame versions of the observational dataset.

    The first version contains all values from simulation. The second
    "missing" (i.e. more realistic version) treats all values for score
    change and potential outcome as missing under treatment T=0. I.e.
    whenever the bank decides not to give the loan, they are passing on a
    potential customer and thus cannot measure the potential outcome or score
    change for that individual.
    """
    # Convert the dataset into a pandas dataframe
    score_changes = dataset.outcomes[:, 0]
    profits = dataset.outcomes[:, 1]
    loan_approved = dataset.outcomes[:, 2]
    repaid = dataset.outcomes[:, 3]

    data = np.concatenate(
        [dataset.covariates,
         score_changes.reshape(-1, 1),
         profits.reshape(-1, 1),
         loan_approved.reshape(-1, 1),
         repaid.reshape(-1, 1)
         ], axis=1)
    columns = dataset.causal_graph.graph["covariate_names"] \
              + ["Score Change", "Profit", "Loan Approved", "Repaid"]
    df = pd.DataFrame(data, columns=columns)
    df.head()

    df_missing = df.copy()

    df_missing['Repaid'][df_missing['Loan Approved'] == 0] = np.nan
    df_missing['Repaid'][df_missing['Loan Approved'] == 0] = np.nan

    return df, df_missing


@gin.configurable
def run_simulation(num_samps: int, policy_name: str, seed: int = None,
                   show_progress: bool = False):
    """Run simulation yielding observational dataset."""
    assert policy_name in ('maxprof', 'dempar', 'eqopp'), 'bad bank policy'
    exper, config, data_args, dynamics, f_T = get_experiment(policy_name)
    dataset = exper.run(num_samples=num_samps, seed=seed, causal_graph=True,
                        show_progress=show_progress, parallelize=False)
    df, df_missing = get_dataframe(dataset)

    return dataset, df, df_missing, config, data_args, dynamics, f_T


def main(unused_argv):
    """Estimate causal params of Liu et al dynamics from observational data."""

    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

    results_dir = gin.query_parameter('%results_dir')
    results_dir = os.path.normpath(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # log stuff to disk
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    logging.get_absl_handler().use_absl_log_file('ensemble_bureau_policy',
                                                 results_dir)

    def log(msg):
        """Log stuff"""
        print(msg)
        logging.info(msg)

    (_, source_df_all, source_df, _, _, _, _) = run_simulation(
        policy_name=SOURCE_POLICY_NAME)
    (_, target_df_all, target_df, _, _, _, _) = run_simulation(
        policy_name=TARGET_POLICY_NAME)

    ############################################################################
    # save data to disk
    data_filename = '%s/data.seed%d.p' % (results_dir,
                                          gin.query_parameter('%seed'))
    data = dict(
        source_df=source_df,
        source_df_all=source_df_all,
        target_df=target_df,
        target_df_all=target_df_all,
    )
    with open(data_filename, 'wb') as f:
        pickle.dump(data, f)

    # Finally, write gin config to disk
    with open(os.path.join(results_dir, 'config.gin'), 'w') as f:
        f.write(gin.operative_config_str())

    log('done. see results at \n%s' % results_dir)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        'gin_file', './config/learn_bureau_policy.gin',
        'Path of config file.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')

    app.run(main)
