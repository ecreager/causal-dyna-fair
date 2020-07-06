"""Run simulation for multiple steps."""

import os
import pickle
import sys
from typing import Dict

from absl import app
from absl import flags
import numpy as np
import gin
import torch
from tqdm import tqdm

from simulation import OneStepSimulation
import structural_eqns as se
from utils.data import get_data_args
from utils.policy import get_policy
from utils.policy import get_dempar_policy_from_selection_rate
from utils.policy import get_eqopp_policy_from_selection_rate

@gin.configurable
def get_simulation(
        utility_repay=gin.REQUIRED,
        utility_default=gin.REQUIRED,
        score_change_repay=gin.REQUIRED,
        score_change_default=gin.REQUIRED):
    """Get a multi-step simulation going."""
    data_args = get_data_args()
    inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, \
            rate_indices = data_args  # pylint: disable=unused-variable
    utils = (utility_default, utility_repay)
    impact = (score_change_default, score_change_repay)
    prob_A_equals_1 = group_size_ratio[-1]
    f_A = se.IndivGroupMembership(prob_A_equals_1)
    f_X = se.InvidScore(*inv_cdfs)
    f_Y = se.RepayPotentialLoan(*loan_repaid_probs)
    f_T = get_policy(loan_repaid_probs, pis, group_size_ratio, utils, impact,
                     scores_list)
    f_Xtilde = se.ScoreUpdate(*impact)
    f_u = se.InstitUtil(*utils)
    f_Umathcal = se.AvgInstitUtil()
    f_Deltaj = se.AvgGroupScoreChange()
    simulation = MultiStepSimulation(
        f_A, f_X, f_Y, f_T, f_Xtilde, f_u, f_Umathcal, f_Deltaj,
        )
    return simulation, data_args, utils, impact


class MultiStepSimulation(OneStepSimulation):
    """Runs simulation for multiple step of dynamics."""

    def run(self, num_steps: int, num_samps: int) -> Dict:
        """Run simulation forward for num_steps and return all observables."""
        blank_tensor = torch.zeros(num_samps)
        A = self.f_A(blank_tensor)
        Xs, Ys, Ts, us, Umathcals = [], [], [], [], []
        Xinit = self.f_X(A)
        X = Xinit
        for _ in range(num_steps):
            Xs.append(X)
            Y = self.f_Y(X, A)
            Ys.append(Y)
            T = self.f_T(X, A)
            Ts.append(T)
            u = self.f_u(Y, T)
            us.append(u)
            Xtilde = self.f_Xtilde(X, Y, T)
            X = Xtilde
            Umathcal = self.f_Umathcal(u)
            Umathcals.append(Umathcal)
        Deltaj = self.f_Deltaj(Xinit, Xtilde, A)  # compute final improvements
        Xs = torch.stack(Xs, dim=0)
        Ys = torch.stack(Ys, dim=0)
        Ts = torch.stack(Ts, dim=0)
        us = torch.stack(us, dim=0)
        Umathcal = torch.mean(torch.stack(Umathcals, dim=0))
        return_dict = dict(
            A=A,
            X=Xs,
            Y=Ys,
            T=Ts,
            u=us,
            Deltaj=Deltaj,
            Umathcal=Umathcal,
            )
        return return_dict


def main(unused_argv):
    """Get multi-step results by sweeping inverventions"""
    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

    seed = gin.query_parameter('%seed')
    results_dir = gin.query_parameter('%results_dir')
    results_dir = os.path.normpath(results_dir)
    num_samps = gin.query_parameter('%num_samps')
    num_steps = gin.query_parameter('%num_steps')

    results_dir = os.path.normpath(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    torch.manual_seed(seed)

    simulation, data_args, _, _ = get_simulation()

    inv_cdfs, loan_repaid_probs, pis, _, scores, \
            rate_indices = data_args
    rate_index_A, rate_index_B = rate_indices

    ############################################################################
    # outcome and utility curves
    ############################################################################

    def check(results):
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    msg = 'NaN spotted in results for variable ' + k
                    raise ValueError(msg)

    outcome_curve_A = []
    outcome_curve_B = []
    utility_curve_A = []
    utility_curve_B = []
    # NOTE: To match fidelity of Liu et al plots we sweep twice, with each group
    #       evaluting results at a different selection_rate grid.
    for selection_rate in tqdm(rate_index_A):
        f_T = get_dempar_policy_from_selection_rate(selection_rate, inv_cdfs)
        simulation.intervene(f_T=f_T)
        results = simulation.run(num_steps, num_samps)
        check(results)
        DeltaA, _ = [mdj.item() for mdj in results['Deltaj']]
        if (results['A'] != 0).all():  # no members of this group
            UmathcalA = 0.
        else:
            batched_A_mask = \
                torch.unsqueeze(results['A'] == 0, 0).repeat(num_steps, 1)
            UmathcalA = torch.mean(results['u'][batched_A_mask]).item()
        outcome_curve_A.append(DeltaA)
        utility_curve_A.append(UmathcalA)

    for selection_rate in tqdm(rate_index_B):
        f_T = get_dempar_policy_from_selection_rate(selection_rate, inv_cdfs)
        simulation.intervene(f_T=f_T)
        results = simulation.run(num_steps, num_samps)
        check(results)
        _, DeltaB = [mdj.item() for mdj in results['Deltaj']]
        if (results['A'] != 1).all():  # no members of this group
            UmathcalB = 0.
        else:
            batched_A_mask = \
                torch.unsqueeze(results['A'] == 1, 0).repeat(num_steps, 1)
            UmathcalB = torch.mean(results['u'][batched_A_mask]).item()
        outcome_curve_B.append(DeltaB)
        utility_curve_B.append(UmathcalB)

    outcome_curve_A = np.array(outcome_curve_A)
    outcome_curve_B = np.array(outcome_curve_B)
    utility_curves = np.array([
        utility_curve_A,
        utility_curve_B,
        ])
    util_MP = np.amax(utility_curves, axis=1)
    utility_curves_MP = np.vstack(
        [utility_curves[0] + util_MP[1], utility_curves[1]+ util_MP[0]]
        )

    # collect DemPar results
    utility_curves_DP = [[], []]
    for i in tqdm(range(len(rate_index_A))):
        beta_A = rate_index_A[i]
        beta_B = rate_index_B[i]
        # get global util results under dempar at selection rate beta_A
        f_T_at_beta_A = get_dempar_policy_from_selection_rate(
            beta_A, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_A)
        results = simulation.run(num_steps, num_samps)
        check(results)
        Umathcal_at_beta_A = results['Umathcal'].item()
        utility_curves_DP[0].append(Umathcal_at_beta_A)
        # get global util results under dempar at selection rate beta_B
        f_T_at_beta_B = get_dempar_policy_from_selection_rate(
            beta_B, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_B)
        results = simulation.run(num_steps, num_samps)
        check(results)
        Umathcal_at_beta_B = results['Umathcal'].item()
        utility_curves_DP[1].append(Umathcal_at_beta_B)
    utility_curves_DP = np.array(utility_curves_DP)

    # collect EqOpp results
    utility_curves_EO = [[], []]
    for i in tqdm(range(len(rate_index_A))):
        beta_A = rate_index_A[i]
        beta_B = rate_index_B[i]
        # get global util results under dempar at selection rate beta_A
        f_T_at_beta_A = get_eqopp_policy_from_selection_rate(
            beta_A, loan_repaid_probs, pis, scores)
        simulation.intervene(f_T=f_T_at_beta_A)
        results = simulation.run(num_steps, num_samps)
        check(results)
        Umathcal_at_beta_A = results['Umathcal'].item()
        utility_curves_EO[0].append(Umathcal_at_beta_A)
        # get global util results under dempar at selection rate beta_B
        f_T_at_beta_B = get_dempar_policy_from_selection_rate(
            beta_B, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_B)
        results = simulation.run(num_steps, num_samps)
        check(results)
        Umathcal_at_beta_B = results['Umathcal'].item()
        utility_curves_EO[1].append(Umathcal_at_beta_B)
    utility_curves_EO = np.array(utility_curves_EO)

    results.update(dict(
        rate_index_A=rate_index_A,
        rate_index_B=rate_index_B,
        outcome_curve_A=outcome_curve_A,
        outcome_curve_B=outcome_curve_B,
        utility_curves_MP=utility_curves_MP,
        utility_curves_DP=utility_curves_DP,
        utility_curves_EO=utility_curves_EO))

    ############################################################################
    # Finally, write commands, script, and results to disk
    ############################################################################
    # for reproducibility, copy command and script contents to results
    if results_dir not in ('.', ):
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
        'gin_file', './config/multi_step_simulation.gin', 'Config file path.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')

    app.run(main)

