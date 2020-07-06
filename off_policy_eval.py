"""Carry out off-policy analysis using previously generated obs. data."""
import argparse
import pickle
import os


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# config
THRESHOLDS = np.arange(300, 850, 5)
AXISFONTSIZE = 20
LEGENDSIZE = 20
TITLESIZE = 20
TICKSIZE = 15


def plot_estimation_errors(args):
    """Plot subfigures needed for figure 5."""
    # loop over thresholds
    all_inds = np.ones(shape=valid_data.shape[0]).astype(bool)
    inds_0 = valid_data[:, 0] < 0.5
    inds_1 = valid_data[:, 0] > 0.5
    scores = valid_data[:, 1]

    # compute estimation errors
    results = {}
    for do_rewards in [True, False]:
        results[do_rewards] = {}
        for outcome_type in ['utilities', 'score_changes']:
            print(outcome_type)
            if outcome_type == 'utilities':
                outcome_models = utility_outcome_models
            else:
                outcome_models = delta_outcome_models

            for inds, nm in [(all_inds, 'all'), (inds_0, '0'), (inds_1, '1')]:
                print(nm)
                scores = valid_data[inds, 1]
                res = {}
                for thr in THRESHOLDS:
                    thr_results = []
                    target_treatments = (scores > thr).astype(float)
                    for i in range(len(train_data)):
                        (true_mean_reward, regression_mean_reward,
                         iw_mean_reward, dr_mean_reward) = evaluate_policy(
                             valid_data, target_treatments, inds,
                             clf_outcome=outcome_models[i],
                             clf_treatment=treatment_models[i],
                             outcome_type=outcome_type, rewards=do_rewards)
                        thr_results.append(
                            (true_mean_reward, regression_mean_reward,
                             iw_mean_reward, dr_mean_reward)
                        )
                    res[thr] = thr_results
                if do_rewards:
                    plt.clf()

                    reg_means = [np.mean([abs(res[thr][i][1] - res[thr][i][0])
                                          for i in range(len(train_data))])
                                 for thr in THRESHOLDS]
                    reg_stds = [np.std([abs(res[thr][i][1] - res[thr][i][0])
                                        for i in range(len(train_data))])
                                for thr in THRESHOLDS]
                    dr_means = [np.mean([abs(res[thr][i][3] - res[thr][i][0])
                                         for i in range(len(train_data))])
                                for thr in THRESHOLDS]
                    dr_stds = [np.std([abs(res[thr][i][3] - res[thr][i][0])
                                       for i in range(len(train_data))])
                               for thr in THRESHOLDS]

                    regcol = 'b'
                    drcol = 'r'
                    plt.fill_between(
                        THRESHOLDS,
                        [reg_means[i] - reg_stds[i] for i in range(len(reg_means))],
                        [reg_means[i] + reg_stds[i] for i in range(len(reg_means))],
                        color=regcol, alpha=0.5
                    )
                    plt.plot(THRESHOLDS, reg_means, label='Regression', c=regcol)
                    plt.fill_between(
                        THRESHOLDS,
                        [dr_means[i] - dr_stds[i] for i in range(len(dr_means))],
                        [dr_means[i] + dr_stds[i] for i in range(len(dr_means))],
                        color=drcol, alpha=0.5
                    )
                    plt.plot(THRESHOLDS, dr_means, label='Doubly Robust', c=drcol)
                    plt.xlabel('Threshold', fontsize=AXISFONTSIZE)
                    plt.ylabel('{} Estimation Error'.format(
                        '$E_{\pi}[u]$' if outcome_type == 'utilities'
                        else '$E_{\pi}[\Delta]$'), fontsize=AXISFONTSIZE
                              )
                    plt.title('Group A={}'.format(nm) if nm in ('0', '1') else nm, fontsize=TITLESIZE)
                    plt.xticks(fontsize=TICKSIZE)
                    plt.yticks(fontsize=TICKSIZE)

                    plt.legend(prop={'size': LEGENDSIZE})
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(args.results_dir,
                                     'threshold_policy_evaluation_{}_group_{}.pdf'
                                     .format(outcome_type, nm)))

                results[do_rewards][(outcome_type, nm)] = res
    inds = {'all': all_inds,
            '0': inds_0,
            '1': inds_1,
           }  # indices denoting positive and negative treatments
    return inds, results


def test_eqopp_thresholds(lmbda, best_thr, test_data):
    """Get true eqopp distance."""
    inds_0 = test_data[:, 0] < 0.5
    inds_1 = test_data[:, 0] > 0.5
    true_eqopp_value_0 = get_true_eqopp_value(test_data, inds_0, best_thr[0])
    true_eqopp_value_1 = get_true_eqopp_value(test_data, inds_1, best_thr[1])
    print('P(Y | T): {:.5f}, {:.5f}'.format(
        true_eqopp_value_0, true_eqopp_value_1))
    true_eqopp_distance = abs(true_eqopp_value_0 - true_eqopp_value_1)
    true_utility = get_true_utility(test_data, best_thr[0], best_thr[1])
    true_obj_value = true_utility - true_eqopp_distance * lmbda
    print('True obj value at chosen thresholds: {:.3f}'.format(
        true_obj_value))
    print('True utility: {:.3f}, True eqopp: {:.5f}'.format(
        true_utility, true_eqopp_distance))
    return (true_utility, true_eqopp_distance)


def plot_policy_objective(args, train_data, valid_data, test_data, results,
                          inds):
    """Plot figure 6."""

    lmbdas = np.arange(0, 1, 0.1)
    res_eqopp = {}

    for clf_i in range(len(train_data)):
        print(clf_i)
        res_eqopp[clf_i] = {}
        thr_cache_res = get_eqopp_threshold_cache(
            1, 'regression', clf_i, valid_data, results, inds)
        thr_cache_dr = get_eqopp_threshold_cache(
            3, 'doubly robust', clf_i, valid_data, results, inds)
        for lmbda in lmbdas:
            print('regression estimator')
            best_thr_reg = get_eqopp_thresholds_from_cache(lmbda, thr_cache_res)
            u_reg, eq_reg = test_eqopp_thresholds(
                lmbda, best_thr_reg, test_data)
            print('doubly robust estimator')
            best_thr_dr = get_eqopp_thresholds_from_cache(lmbda, thr_cache_dr)
            u_dr, eq_dr = test_eqopp_thresholds(lmbda, best_thr_dr, test_data)
            res_eqopp[clf_i][lmbda] = {
                'reg': (u_reg, eq_reg), 'dr': (u_dr, eq_dr)
            }

    lw = 3
    plt.clf()
    regcol = 'b'
    drcol = 'r'
    reg_means = np.array([np.mean([res_eqopp[clf_i][lmbda]['reg'][0]
                                   - lmbda * res_eqopp[clf_i][lmbda]['reg'][1]
                                   for clf_i in range(len(train_data))])
                          for lmbda in lmbdas])
    dr_means = np.array([np.mean([res_eqopp[clf_i][lmbda]['dr'][0]
                                  - lmbda * res_eqopp[clf_i][lmbda]['dr'][1]
                                  for clf_i in range(len(train_data))])
                         for lmbda in lmbdas])
    reg_stds = np.array([np.std([res_eqopp[clf_i][lmbda]['reg'][0]
                                 - lmbda * res_eqopp[clf_i][lmbda]['reg'][1]
                                 for clf_i in range(len(train_data))])
                         for lmbda in lmbdas])
    dr_stds = np.array([np.std([res_eqopp[clf_i][lmbda]['dr'][0]
                                - lmbda * res_eqopp[clf_i][lmbda]['dr'][1]
                                for clf_i in range(len(train_data))])
                        for lmbda in lmbdas])

    plt.fill_between(lmbdas,
                     [reg_means[i] - reg_stds[i] for i in range(len(reg_means))],
                     [reg_means[i] + reg_stds[i] for i in range(len(reg_means))],
                     color=regcol, alpha=0.3)
    plt.plot(lmbdas, reg_means,
             label='Regression', lw=lw)
    plt.fill_between(lmbdas,
                     [dr_means[i] - dr_stds[i] for i in range(len(reg_means))],
                     [dr_means[i] + dr_stds[i] for i in range(len(reg_means))],
                     color=drcol, alpha=0.3)
    plt.plot(lmbdas, dr_means,
             label='Doubly Robust', lw=lw)
    plt.xlabel('$\lambda$', fontsize=AXISFONTSIZE)
    plt.ylabel('$\mathcal{V}_{\pi}$', fontsize=AXISFONTSIZE)
    plt.xticks(fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.legend(prop={'size': LEGENDSIZE})
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'policy_objective.pdf'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Off-policy analysis.')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed for the analysis.')
    parser.add_argument('--obs_data_dir', type=str,
                        default='/tmp/causal-dyna-fair/observational_data',
                        help='Directory containing observational data.')
    parser.add_argument('--results_dir', type=str,
                        default='/tmp/causal-dyna-fair/off_policy_eval',
                        help='Directory where plots should be saved.')
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    np.random.seed(args.seed)

    # Load, process, and split data
    NSEEDS = 13  # number of observational datasets previously generated
    data_filenames = [
        os.path.join(args.obs_data_dir, 'data.seed{:d}.p'.format(seed))
        for seed in range(NSEEDS)
    ]
    data_files = []
    for data_filename in data_filenames:
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)
            data_files.append(data)
            # The keys of the pickle dict are
            # * 'source_df' - pd.DataFrame from max profit policy
            #                 with missing values removed
            # * 'source_df_all' - pd.DataFrame from max profit policy
            #                     with _all_ values included
            #
            # For each pd.DataFrame, the columns are
            # 'group_0': 1 if race == 'White' else 0
            # 'credit_score_0': initial credit score
            # 'Score Change': difference in final and initial credit score
            # 'Profit': bank profit for this individual
            # 'Loan Approved': whether loan was offered
            # 'Loan Repaid': whether loan was repaid

    def process_dataframe(df, flip):
        cols = df['source_df_all'].columns
        indices = {cols[i]: i for i in range(len(cols))}
        d = df['source_df_all'].values
        noise = np.random.binomial(1, 1 - flip, size=d.shape[0])
        loan_i = indices['Loan Approved']
        noisy_treatments = noise * d[:, loan_i] + (1 - noise) * (
            1 - d[:, loan_i]
        )
        dn = np.copy(d)
        dn[:, loan_i] = noisy_treatments
        return dn

    flip_prob = 0.1  # need this to satisfy overlap assumption
    n_train_data = 11  # 11 dsets used to train
    valid_data_ix = 11  # 12th used for validation data
    test_data_ix = 12  # 13th is test set
    train_data = [process_dataframe(d, flip_prob)
                  for d in data_files[:n_train_data]]
    valid_data = process_dataframe(data_files[valid_data_ix], flip_prob)
    test_data = process_dataframe(data_files[test_data_ix], flip_prob)


    def train_outcome_model(d, outcome_type):
        assert outcome_type in ('utilities', 'score_changes')
        received_treatment = d[:, 4].astype(bool)
        outcomes = d[received_treatment, 5]
        clf_outcome = LogisticRegression(solver='liblinear').fit(
            d[received_treatment, :2], outcomes
        )
        return clf_outcome

    def train_treatment_model(d):
        clf_treatment = LogisticRegression(solver='liblinear').fit(
            d[:, :2], d[:, 4]
        )
        return clf_treatment

    utility_outcome_models = [train_outcome_model(tr, 'utilities')
                              for tr in train_data]
    delta_outcome_models = [train_outcome_model(tr, 'score_changes')
                            for tr in train_data]
    treatment_models = [train_treatment_model(tr) for tr in train_data]


    def evaluate_policy(d, target_treatments, inds,
                        clf_outcome, clf_treatment, outcome_type='utilities',
                        rewards=True):
        assert outcome_type in ('utilities', 'score_changes')

        treatments = d[inds, 4]
        all_user_data = d[inds, :2]
        if outcome_type == 'utilities':
            true_outcomes = d[inds, 5]
            if rewards:
                true_utilities = true_outcomes - 4 * (1 - true_outcomes)
            else:
                true_utilities = true_outcomes
        else:
            true_outcomes = d[inds, 5]
            if rewards:
                true_utilities = true_outcomes * 75 - 150 * (1 - true_outcomes)
            else:
                true_utilities = true_outcomes

        # true reward of policy
        true_mean_reward = np.mean(target_treatments * true_utilities)

        # regression estimate of policy reward
        predicted_outcomes = clf_outcome.predict_proba(all_user_data)
        if outcome_type == 'utilities':
            if rewards:
                predicted_utilities = predicted_outcomes[:, 1] \
                    - 4 * predicted_outcomes[:, 0]
            else:
                predicted_utilities = predicted_outcomes[:, 1]
        else:
            if rewards:
                predicted_utilities = 75 * predicted_outcomes[:, 1] \
                    - 150 * predicted_outcomes[:, 0]
            else:
                predicted_utilities = predicted_outcomes[:, 1]
        regression_reward = target_treatments * predicted_utilities
        regression_mean_reward = np.mean(regression_reward)

        # inverse probability weighting estimate of policy value
        C = np.equal(target_treatments.astype(int), treatments.astype(int))
        predicted_treatments = clf_treatment.predict_proba(all_user_data)[:, 1]
        predicted_C = predicted_treatments * target_treatments + (
            1 - predicted_treatments
        ) * (1 - target_treatments)

        observed_reward = true_utilities * treatments
        weighted_values = ((C * observed_reward) / predicted_C)
        iw_mean_reward = np.mean(weighted_values)

        # doubly robust estimate of policy value
        dr_value = weighted_values - (
            (C - predicted_C) / predicted_C
        ) * regression_reward
        dr_mean_reward = np.mean(dr_value)

        return (true_mean_reward,
                regression_mean_reward,
                iw_mean_reward,
                dr_mean_reward)


    inds, results = plot_estimation_errors(args)


    def estimate_p_y_1(data, inds, outcome_type, estimator_index, clf_i):
        if outcome_type == 'utilities':
            outcome_models = utility_outcome_models
        else:
            outcome_models = delta_outcome_models

        p = 0.9
        nrounds = 20

        estimates = []
        for _ in range(nrounds):
            target_treatments = np.random.binomial(1, p, size=inds.sum())
            eval_results = evaluate_policy(
                data, target_treatments, inds,
                clf_outcome=outcome_models[clf_i],
                clf_treatment=treatment_models[clf_i],
                outcome_type=outcome_type, rewards=False
            )
            estimates.append(eval_results[estimator_index] * (1 / p))
        return np.mean(estimates)

    p_y_1_vals = {}

    for nm_, inds_ in inds.items():
        nm_ = str(nm_)
        for est_index in [0, 1, 3]:
            for outcome_type in ['utilities', 'score_changes']:
                print(nm_, est_index, outcome_type)
                p_y_1_vals[(outcome_type, nm_, est_index)] = []
                for clf_i in range(len(train_data)):
                    mn = np.mean(estimate_p_y_1(
                        valid_data, inds_, outcome_type, est_index, clf_i))
                    p_y_1_vals[(outcome_type, nm_, est_index)].append(mn)


    def get_eqopp_value(data, inds, thr, group_name, estimator_index,
                        clf_i, results):
        # want P(T = 1 | Y_1 = 1) = P(Y = 1 | T = 1) * P(T = 1) / P(Y = 1)
        scores = data[inds, 1]
        target_treatments = (scores > thr).astype(float)
        p_t_1 = np.mean(target_treatments)
        p_y_1 = p_y_1_vals[('score_changes',
                            group_name,
                            estimator_index)][clf_i]
        e_y_1_obs = np.mean(
            [x[estimator_index]
             for x in results[False][('score_changes', group_name)][thr]
            ]
        )
        EPS = 1e-7  # for numerical stability
        p_y_1_given_t_1 = e_y_1_obs * np.sum(inds) / (
            np.sum(target_treatments) + EPS
        )
        p_t_1_given_y_1 = p_y_1_given_t_1 * p_t_1 / p_y_1
        return p_t_1_given_y_1

    def get_utility(t0, t1, estimator_index, clf_i, results, inds):
        utility_0 = results[True][('utilities', '0')][t0][clf_i]
        utility_1 = results[True][('utilities', '1')][t1][clf_i]
        utility = utility_0[estimator_index] * np.mean(
            inds['0']) + utility_1[estimator_index] * np.mean(inds['1'])
        return utility

    def get_true_eqopp_value(data, inds, thr):
        outcomes = data[inds, 5]
        scores = data[inds, 1]
        target_treatments = (scores > thr).astype(float)
        p_t_1_given_y_1 = np.mean(target_treatments[outcomes > 0.5])
        return p_t_1_given_y_1

    def calculate_total_true_utility(data, thr, inds):
        scores = data[inds, 1]
        trmt = (scores > thr).astype(float)
        outcomes = data[inds, 5]
        utility = np.sum(outcomes * trmt) - 4 * np.sum((1 - outcomes) * trmt)
        return utility

    def get_true_utility(data, t0, t1):
        """Get total utility for each group."""
        inds_0 = data[:, 0] < 0.5
        utility_0 = calculate_total_true_utility(data, t0, inds_0)
        inds_1 = data[:, 0] > 0.5
        utility_1 = calculate_total_true_utility(data, t1, inds_1)
        return (utility_0 + utility_1) / data.shape[0]

    def get_eqopp_threshold_cache(estimator_index, estimator_name, clf_i,
                                  valid_data, results, inds):
        """Find equal opportunity thresholds acc for each estimator."""
        print('caching results for {} estimator'.format(estimator_name))
        thr_cache = {}
        inds_0 = valid_data[:, 0] < 0.5
        inds_1 = valid_data[:, 0] > 0.5
        for t1 in THRESHOLDS:
            eqopp_value_1 = get_eqopp_value(
                valid_data, inds_1, t1, '1', estimator_index, clf_i, results)
            for t0 in THRESHOLDS:
                eqopp_value_0 = get_eqopp_value(
                    valid_data, inds_0, t0, '0', estimator_index, clf_i,
                    results)
                eqopp_distance = abs(eqopp_value_0 - eqopp_value_1)
                utility = get_utility(t0, t1, estimator_index,
                                      clf_i, results, inds)
                thr_cache[(t0, t1)] = (utility, eqopp_distance)
        return thr_cache

    def get_eqopp_thresholds_from_cache(lmbda, cache):
        """Find equal opportunity thresholds acc for each estimator,"""
        max_obj_value = -99999
        best_thr = None
        best_utility = None
        best_eqopp = None
        for t0 in THRESHOLDS:
            for t1 in THRESHOLDS:
                utility, eqopp_distance = cache[(t0, t1)]
                obj_value = utility - eqopp_distance * lmbda
                if obj_value > max_obj_value:
                    max_obj_value = obj_value
                    best_thr = (t0, t1)
                    best_utility = utility
                    best_eqopp = eqopp_distance
        print('Improved obj value to {:.3f} at ({:d}, {:d})'
              .format(max_obj_value, int(best_thr[0]), int(best_thr[1])))
        print('Utility: {:.3f}, Eqopp: {:.3f}'.format(
            best_utility, best_eqopp))
        return best_thr

    plot_policy_objective(
        args, train_data, valid_data, test_data, results, inds)
