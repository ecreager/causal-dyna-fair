"""Utilities for initializing policies as StructuralEqns."""
import gin
import numpy as np
import pandas as pd

import distribution_to_loans_outcomes as dlo
from structural_eqns import ThresholdLoanPolicy
from utils.data import get_inv_cdf_fns

@gin.configurable
def get_policy(loan_repaid_probs,
               pis,
               group_size_ratio,
               utils,
               score_change_fns,
               scores,
               policy_name=gin.REQUIRED):
    """Get named threshold policy StructuralEqn."""

    # Threshold policies maximizing profit possibly under a fairness constraint
    thresh_dempar, thresh_eqopp, thresh_maxprof, thresh_downwards = \
        dlo.get_thresholds(loan_repaid_probs, pis, group_size_ratio, utils,
                           score_change_fns, scores)
    del thresh_downwards  # unused

    if policy_name.lower() == 'maxprof':
        return ThresholdLoanPolicy(*thresh_maxprof)  # pylint: disable=no-value-for-parameter

    if policy_name.lower() == 'dempar':
        return ThresholdLoanPolicy(*thresh_dempar)  # pylint: disable=no-value-for-parameter

    if policy_name.lower() == 'eqopp':
        return ThresholdLoanPolicy(*thresh_eqopp)  # pylint: disable=no-value-for-parameter

    raise ValueError('Bad policy name: {}'.format(policy_name))

def get_dempar_policy_from_selection_rate(selection_rate, inv_cdfs):
    """Return policy that achieves specified selection rate for both groups."""
    threshes = [inv_cdf(1. - selection_rate) for inv_cdf in inv_cdfs]
    return ThresholdLoanPolicy(*threshes)  # pylint: disable=no-value-for-parameter

def get_eqopp_policy_from_selection_rate(selection_rate,
                                         loan_repaid_probs,
                                         pis,
                                         scores):
    """Return policy with specified selection rate on Y=1 for both groups."""
    # compute P(X, Y=1|A)
    p_X_Yeq1_cond_A = np.array([
        [
            pis[j, i] * loan_repaid_probs[j](X)  # P(X|A) * P(Y=1|X,A)
            for i, X in enumerate(scores)
        ] for j in range(2)
    ])
    # compute P(X|Y=1, A)
    p_X_cond_Yeq1_A = \
        p_X_Yeq1_cond_A / p_X_Yeq1_cond_A.sum(axis=1, keepdims=True)
    cdfs = np.cumsum(p_X_cond_Yeq1_A, axis=1)
    cdfs = pd.DataFrame(data=cdfs.T, index=scores, columns=('Black', 'White'))
    new_inv_cdfs = get_inv_cdf_fns(cdfs)
    return get_dempar_policy_from_selection_rate(selection_rate, new_inv_cdfs)
