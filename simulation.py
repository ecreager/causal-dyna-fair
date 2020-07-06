"""Single simulation of one-step FICO dynamics under Liu et al 2018 SCM."""

import os
import pickle
import sys
from typing import Dict

from absl import app
from absl import flags
import gin
import torch

import structural_eqns as se
from utils.policy import get_policy
from utils.data import get_data_args


class OneStepSimulation:
    """Runs simulation for one step of dynamics under Liu et al 2018 SCM."""
    def __init__(self,
                 f_A: se.StructuralEqn,  # stochastic SE for group membership
                 f_X: se.StructuralEqn,  # stochastic SE for indiv scores
                 f_Y: se.StructuralEqn,  # stochastic SE for potential repayment
                 f_T: se.StructuralEqn,  # SE for threshold loan policy
                 f_Xtilde: se.StructuralEqn,  # SE for indiv score change
                 f_u: se.StructuralEqn,  # SE for individual utility
                 f_Umathcal: se.StructuralEqn,  # SE for avg instit. utility
                 f_Deltaj: se.StructuralEqn,  # SE per-group avg score change
                 ) -> None:
        self.f_A = f_A
        self.f_X = f_X
        self.f_Y = f_Y
        self.f_T = f_T
        self.f_Xtilde = f_Xtilde
        self.f_u = f_u
        self.f_Deltaj = f_Deltaj
        self.f_Umathcal = f_Umathcal

    def run(self, num_steps: int, num_samps: int) -> Dict:
        """Run simulation forward for num_steps and return all observables."""
        if num_steps != 1:
            raise ValueError('Only one-step dynamics are currently supported.')
        blank_tensor = torch.zeros(num_samps)
        A = self.f_A(blank_tensor)
        X = self.f_X(A)
        Y = self.f_Y(X, A)
        T = self.f_T(X, A)
        Xtilde = self.f_Xtilde(X, Y, T)
        u = self.f_u(Y, T)
        Deltaj = self.f_Deltaj(X, Xtilde, A)
        Umathcal = self.f_Umathcal(u)
        return_dict = dict(
            A=A,
            X=X,
            Y=Y,
            T=T,
            u=u,
            Xtilde=Xtilde,
            Deltaj=Deltaj,
            Umathcal=Umathcal,
            )
        return return_dict

    def intervene(self, **kwargs):
        """Update attributes via intervention."""
        for k, v in kwargs.items():
            setattr(self, k, v)

def main(unused_argv):
    """Produces figures from Liu et al 2018 and save results."""
    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

    seed = gin.query_parameter('%seed')
    results_dir = gin.query_parameter('%results_dir')
    results_dir = os.path.normpath(results_dir)
    num_steps = gin.query_parameter('%num_steps')
    num_samps = gin.query_parameter('%num_samps')
    utility_repay = gin.query_parameter('%utility_repay')
    utility_default = gin.query_parameter('%utility_default')
    score_change_repay = gin.query_parameter('%score_change_repay')
    score_change_default = gin.query_parameter('%score_change_default')

    torch.manual_seed(seed)

    inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, _ = \
            get_data_args()
#    import pdb
#    pdb.set_trace()
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

    simulation = OneStepSimulation(
        f_A, f_X, f_Y, f_T, f_Xtilde, f_u, f_Umathcal, f_Deltaj,
        )

    results = simulation.run(num_steps, num_samps)

    # add thresholds determined by solver
    policy_name = gin.query_parameter('%policy_name')
    situation = 'situation1' if (utility_default == -4) else 'situation2'
    these_thresholds = {
        situation:
        {policy_name: [f_T.threshold_group_0, f_T.threshold_group_1]}
    }
    results['threshes'] = these_thresholds

   # Finally, write results to disk
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # for reproducibility, copy command and script contents to results
    if results_dir not in ('.', ):
        cmd = 'python ' + ' '.join(sys.argv)
        with open(os.path.join(results_dir, 'command.sh'), 'w') as f:
            f.write(cmd)
        file_basename = os.path.basename(__file__)
        this_script = open(__file__, 'r').readlines()
        with open(os.path.join(results_dir, file_basename), 'w') as f:
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
        'gin_file', './config/one-quarter.gin', 'Path of config file.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')

    app.run(main)

