"""Intervene by sampling Y from P(Y|X) rather than P(Y|X,A) & run multi-step."""
import os
import pickle
import sys

from absl import app
from absl import flags
import gin
import torch

from multi_step_simulation import get_simulation
import structural_eqns as se
from utils.data import  get_marginal_loan_repaid_probs
from utils.policy import get_policy


@gin.configurable
def get_intervened_simulation(
        repayment_intervention=gin.REQUIRED,
        policy_intervention=gin.REQUIRED):
    """Get a multi-step simulation with optional interventions."""
    simulation, data_args, utils, impact = get_simulation()
    _, _, pis, group_size_ratio, scores, _ = data_args
    marginal_loan_repaid_probs = get_marginal_loan_repaid_probs()
    if policy_intervention:
        f_T_marginal = get_policy(marginal_loan_repaid_probs, pis,
                                  group_size_ratio, utils, impact, scores)
        simulation.intervene(f_T=f_T_marginal)
    if repayment_intervention:
        f_Y_marginal = se.RepayPotentialLoan(*marginal_loan_repaid_probs)
        simulation.intervene(f_Y=f_Y_marginal)
    return simulation

@gin.configurable
def query_parameters(
        num_steps=gin.REQUIRED,
        num_samps=gin.REQUIRED,
        seed=gin.REQUIRED,
        results_dir=gin.REQUIRED
        ):
    """Returns leftover gin config parameters."""
    return num_steps, num_samps, seed, results_dir


def main(unused_argv):
    """Get results by sweeping inverventions"""
    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

    num_steps, num_samps, seed, results_dir = query_parameters()

    results_dir = os.path.normpath(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    torch.manual_seed(seed)

    simulation = get_intervened_simulation()

    ############################################################################
    # Outcome and utility curves
    ############################################################################
    def check(results):
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    msg = 'NaN spotted in results for variable ' + k
                    raise ValueError(msg)

    results = simulation.run(num_steps, num_samps)
    check(results)

    ############################################################################
    # Finally, write commands, script, and results to disk
    ############################################################################
    # for reproducibility, copy command and script contents to results
    #DEFAULT_RESULTS_DIR = '/scratch/gobi1/creager/delayedimpact'
    DEFAULT_RESULTS_DIR = '/tmp/delayedimpact'
    if results_dir not in ('.', 'results/python', DEFAULT_RESULTS_DIR):
        cmd = 'python ' + ' '.join(sys.argv)
        with open(os.path.join(results_dir, 'command.sh'), 'w') as f:
            f.write(cmd)
        this_script = open(__file__, 'r').readlines()
        with open(os.path.join(results_dir, __file__), 'w') as f:
            f.write(''.join(this_script))

    results_filename = os.path.join(results_dir, 'results.p')
    with open(results_filename, 'wb') as f:
        _ = pickle.dump(results, f)

    # Finally, write gin config to disk
    with open(os.path.join(results_dir, 'config.gin'), 'w') as f:
        f.write(gin.operative_config_str())


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        'gin_file', './config/multi_step_repayment_intervention.gin',
        'Path of config file.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')

    app.run(main)
