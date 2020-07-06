"""Collect and plot reuslts from multi_step_repayment_intervention.py"""

import glob
import os
import pickle
from pprint import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('mathtext', fontset='stix')

from absl import app
from absl import flags
import numpy as np

FONTSIZE = 40
LINEWIDTH = 4.0
FIGSIZE = (8, 6)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'expdir',
    '/scratch/gobi1/creager/delayedimpact',
    'Experiment root directory.')
flags.DEFINE_string(
    'baseline_glob',
    'multi_step_repayment_intervention.baseline.*',
    'Glob pattern for all results_dirs in the baseline')
flags.DEFINE_string(
    'policy_intervention_glob',
    'multi_step_repayment_intervention.policy-intervention.*',
    'Glob pattern for all results_dirs in the policy intervention')
flags.DEFINE_string(
    'repayment_intervention_glob',
    'multi_step_repayment_intervention.repayment-intervention.*',
    'Glob pattern for all results_dirs in the repayment intervention')

def get_num_steps(config_filename):
    """Get num_steps as int from config file."""
    with open(config_filename, 'r') as f:
        s = f.read()
    num_steps = s.split('query_parameters.num_steps = ')[-1].split('\n')[0]
    num_steps = int(num_steps)
    return num_steps

def main(unused_argv):
    """Load results and plot errors over time."""
    results_dirs = dict()
    try:
        baseline_glob = os.path.join(FLAGS.expdir, FLAGS.baseline_glob)
        policy_intervention_glob = os.path.join(FLAGS.expdir,
                                                FLAGS.policy_intervention_glob)
        repayment_intervention_glob = os.path.join(
            FLAGS.expdir, FLAGS.repayment_intervention_glob)
        results_dirs.update(baseline=glob.glob(baseline_glob))
        results_dirs.update(
            policy_intervention=glob.glob(policy_intervention_glob)
        )
        results_dirs.update(
            repayment_intervention=glob.glob(repayment_intervention_glob)
        )
    except:
        raise ValueError('Bad glob patterns.')
    pprint(results_dirs)
    max_num_steps = len(results_dirs['baseline']) + 1
    Umathcal = {k: np.zeros(max_num_steps) for k in results_dirs.keys()}
    Deltaj = {k: np.zeros((max_num_steps, 2)) for k in results_dirs.keys()}
    config_filename = \
            os.path.join(results_dirs['baseline'][0], 'config.gin')
    for k, v in results_dirs.items():
        for vv in v:
            config_filename = os.path.join(vv, 'config.gin')
            num_steps = get_num_steps(config_filename)
            results_filename = os.path.join(vv, 'results.p')
            with open(results_filename, 'rb') as f:
                results = pickle.load(f)
            Umathcal[k][num_steps] = results['Umathcal'].item()
            Deltaj[k][num_steps, :] = np.array(results['Deltaj'])
    pprint(Umathcal)
    pprint(Deltaj)

    # Make plot
    def get_ax():
        _, ax = plt.subplots(figsize=FIGSIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(0, LineStyle='-', color='grey', alpha=0.4)
        return ax

    # institutional utility
    ax = get_ax()
    keys = ('policy_intervention', 'repayment_intervention')
    label_policy_intervention = (
        '$do(f_T \\rightarrow \hat f^{EO}_T)$'
        )
    label_repayment_intervention = (
        '$do(f_Y \\rightarrow \hat f_Y)$'
        )
    colors = ('b', 'r')
    labels = (label_policy_intervention, label_repayment_intervention)
    for k, l, c in zip(keys, labels, colors):
        err = np.abs(Umathcal[k] - Umathcal['baseline'])
        ax.plot(err, label=l, color=c, linewidth=LINEWIDTH)
    ax.set_xlabel('num steps', fontsize=FONTSIZE)
    ylabel = 'Institutional profit error'
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(loc='lower right', prop=dict(size=int(FONTSIZE / 1.8)))
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE // 2)
    ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE // 2)

    filename_a = os.path.join(FLAGS.expdir,
                              'multi-step-repayment-intervention-a.pdf')
    plt.tight_layout()
    plt.savefig(filename_a)
    plt.close()

    # group mobility
    ax = get_ax()
    keys = ('policy_intervention', 'repayment_intervention')

    label_policy_intervention_0 = (
        '$do(f_T \\rightarrow \hat f^{EO}_T), A=1$'
        )
    kwargs_policy_intervention_0 = dict(color='b',
                                        linestyle='-',
                                        linewidth=LINEWIDTH)

    label_policy_intervention_1 = (
        '$do(f_T \\rightarrow \hat f^{EO}_T), A=0$'
        )
    kwargs_policy_intervention_1 = dict(color='b',
                                        linestyle='--',
                                        linewidth=LINEWIDTH)

    label_repayment_intervention_0 = (
        '$do(f_Y \\rightarrow \hat f_Y), A=1$'
        )
    kwargs_repayment_intervention_0 = dict(color='r',
                                           linestyle='-',
                                           linewidth=LINEWIDTH)

    label_repayment_intervention_1 = (
        '$do(f_Y \\rightarrow \hat f_Y), A=0$'
        )
    kwargs_repayment_intervention_1 = dict(color='r',
                                           linestyle='--',
                                           linewidth=LINEWIDTH)
    all_labels = (
        (label_policy_intervention_1, label_policy_intervention_0),
        (label_repayment_intervention_1, label_repayment_intervention_0),
        )
    all_kwargs = (
        (kwargs_policy_intervention_1, kwargs_policy_intervention_0),
        (kwargs_repayment_intervention_1, kwargs_repayment_intervention_0),
        )

    for k, labels, kwargs in zip(keys, all_labels, all_kwargs):
        for j, (l, kw) in enumerate(zip(labels, kwargs)):
            err = np.abs(Deltaj[k][:, j] - Deltaj['baseline'][:, j])
            ax.plot(err, label=l, **kw)
    ax.set_xlabel('num steps', fontsize=FONTSIZE)
    ylabel = 'Avg score change error'
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(loc='right', prop=dict(size=int(FONTSIZE / 1.8)))
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE // 2)
    ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE // 2)

    filename_b = os.path.join(FLAGS.expdir,
                              'multi-step-repayment-intervention-b.pdf')
    plt.tight_layout()
    plt.savefig(filename_b)
    plt.close()

    
if __name__ == "__main__":
    app.run(main)
