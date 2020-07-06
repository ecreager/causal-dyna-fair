"""Structural equations that capture individual dynamical steps."""

from abc import ABC, abstractmethod

import gin
import torch
from torch import nn


class StructuralEqn(ABC):
    """Base class for structural equations."""

    @abstractmethod
    def sample_exogenous_noise(self, num_samps):
        """Sample U_X where X represents this node."""
        raise NotImplementedError

    @abstractmethod
    def compute_output(self, exogenous_noise, *inputs):
        """Compute X = f_X(U_X Pa(X)) where Pa(X) are the parents of X."""
        raise NotImplementedError

    def forward(self, *inputs):
        """Compute output of this structural eqn."""
        if inputs:
            num_samps = len(inputs[0])
        U = self.sample_exogenous_noise(num_samps)
        return self.compute_output(U, *inputs)

    def __call__(self, *inputs):
        return self.forward(*inputs)


class ScoreUpdate(StructuralEqn):
    """Individual change in score under Liu et al 2018 SCM.

    Xtilde = f_Xtilde(X, Y, T)

    The individual's score {moves up by score_change_repay    }  if Y = 1, T = 1
                           {moves down by score_change_default}  if Y = 0, T = 1
                           {does not move                     }  else
    """

    min_score = 300.
    max_score = 850.

    def __init__(self, score_change_default, score_change_repay):
        self.score_change_default = float(score_change_default)
        self.score_change_repay = float(score_change_repay)

    def sample_exogenous_noise(self, num_samps):
        pass

    def compute_output(self, exogenous_noise, X, Y, T):  # pylint: disable=arguments-differ
        del exogenous_noise  # output is deterministic
        Y = Y.float()
        T = T.float()
        score_change = self.score_change_repay * Y.float() * T + \
                 self.score_change_default * (1. - Y.float()) * T
        X_next_step = X + score_change
        # score limits
        X_next_step = torch.clamp(X_next_step, self.min_score, self.max_score)
        return X_next_step


class InstitUtil(StructuralEqn):
    """Institutional utility due to an individual under Liu et al 2018 SCM.

    u = f_u(Y, T)

    The instit. utility {is utility_repay  }  if Y = 1, T = 1
                        {is utility_default}  if Y = 0, T = 1
                        {is zero           }  else
    """

    def __init__(self, utility_default, utility_repay):
        self.utility_default = float(utility_default)
        self.utility_repay = float(utility_repay)

    def sample_exogenous_noise(self, num_samps):
        pass

    def compute_output(self, exogenous_noise, Y, T):  # pylint: disable=arguments-differ
        del exogenous_noise  # output is deterministic
        T = T.float()
        output = self.utility_repay * Y.float() * T + \
            self.utility_default * (1. - Y.float()) * T
        return output


class BernoulliStructuralEqn(StructuralEqn):
    """Structural equation for a Bernoulli random variable.

    SE output is Bernoulli distributed with parameter computed as a function
    of the inputs. Implemented using the Gumbel-Max reparameterization.
    """
    uniform_sampler = torch.distributions.Uniform(0., 1.)

    def bernoulli_parameter_fn(self, *inputs):
        """Maps inputs to Bernoulli parameter."""
        raise NotImplementedError

    def sample_exogenous_noise(self, num_samps):
        return self.uniform_sampler.sample((num_samps, ))

    def compute_output(self, exogenous_noise, *inputs):
        bernoulli_parameter = self.bernoulli_parameter_fn(*inputs)
        #log = torch.log
        EPS = 1e-8
        log = lambda x: torch.log(torch.clamp(x, EPS, 1.))  # numer. stable log
        output = ((
            log(bernoulli_parameter) - log(1. - bernoulli_parameter) +
            log(exogenous_noise) - log(1. - exogenous_noise))
                  > 0.0).int()  # Gumbel-max trick
        return output


class IndivGroupMembership(BernoulliStructuralEqn):
    """Samples group membership A=1."""

    def __init__(self, prob_A_equals_1):
        self.prob_A_equals_1 = torch.tensor([prob_A_equals_1])

    def bernoulli_parameter_fn(self, *inputs):
        return self.prob_A_equals_1


class InvidScore(StructuralEqn):
    """Samples score X given group membership A."""

    def __init__(self, cdf_X_group_0, cdf_X_group_1):
        """Args are CDFs over scores for each group."""
        self.cdf_X_group_0 = cdf_X_group_0
        self.cdf_X_group_1 = cdf_X_group_1

    def sample_exogenous_noise(self, num_samps):
        return torch.distributions.Uniform(0., 1.).sample((num_samps, ))

    def compute_output(self, exogenous_noise, A):  # pylint: disable=arguments-differ
        A = A.float()
        exogenous_noise = exogenous_noise.numpy()
        output = self.cdf_X_group_1(exogenous_noise) ** A \
                * self.cdf_X_group_0(exogenous_noise) ** (1. - A)
        return output


class RepayPotentialLoan(BernoulliStructuralEqn):
    """Sample repayment potential of loan (if one were given).

    Potential repayment is Bernoulli distributed with probability pi_j(X), where
    X is the individual's score and A=j is their group membership. I.e. we
    sample Y ~ P(Y|X,A)
    """

    def __init__(self,
                 prob_repayment_given_group_0,
                 prob_repayment_given_group_1):
        """Each arg maps score to a prob. of repayment for a different group."""
        self.prob_repayment_given_group_0 = prob_repayment_given_group_0
        self.prob_repayment_given_group_1 = prob_repayment_given_group_1

    def bernoulli_parameter_fn(self, X, A):  # pylint: disable=arguments-differ
        A = A.float()
        X = X.numpy()
        output = self.prob_repayment_given_group_1(X) ** A \
                * self.prob_repayment_given_group_0(X) ** (1. - A)
        return output


class RepayPotentialLoanGroupBlind(BernoulliStructuralEqn):
    """Sample repayment potential of loan (if one were given).

    In this case we sample Y ~ P(Y|X), which we denote as group-blind b/c A is
    not taken into direct consideration.
    """

    def __init__(self, prob_repayment, *args):
        """prob_repayment maps score to a prob. of repayment."""
        del args
        self.prob_repayment = prob_repayment

    def bernoulli_parameter_fn(self, X):  # pylint: disable=arguments-differ
        X = X.numpy()
        output = self.prob_repayment(X)
        output = torch.tensor(output, dtype=torch.float32)
        return output


class Binarize(torch.autograd.Function):
    """Straight-through estimator."""
    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        x = torch.sign(x)+1
        x /= 2
        return x

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        ## Adapt this to account for shift and division
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x > 1] = 0
        grad_x[x < -1] = 0
        return grad_x


class ThresholdLoanPolicy(StructuralEqn):
    """Sample whether loan is offered to indiv with score X from group A.

    This implementation of thresholding is deterministic, i.e., no tie-breaking.
    """

    def __init__(self, threshold_group_0, threshold_group_1):
        self.threshold_group_0 = threshold_group_0
        self.threshold_group_1 = threshold_group_1

    def sample_exogenous_noise(self, num_samps):
        pass

    def compute_output(self, exogenous_noise, X, A):  # pylint: disable=arguments-differ
        def _switch_codes(x):
            """Maps from {-1, 1} to {0, 1}"""
            return (x + 1.) / 2
        output = (
            Binarize.apply(X - self.threshold_group_1) ** A.float()
            * Binarize.apply(X - self.threshold_group_0) ** (1. - A.float())
        )
        return output


class AvgGroupScoreChange(StructuralEqn):
    """Compute average score change per group."""

    def sample_exogenous_noise(self, num_samps):
        pass

    def compute_output(self, exogenous_noise, Xs, Xtildes, As):  # pylint: disable=arguments-differ
        """Compute avg score change per group

        Assumes Xs, Xtildes, and As are batches of observations."""
        group_0_avg_delta = torch.mean(Xtildes[As == 0] - Xs[As == 0])
        group_1_avg_delta = torch.mean(Xtildes[As == 1] - Xs[As == 1])
        return group_0_avg_delta, group_1_avg_delta


class AvgInstitUtil(StructuralEqn):
    """Compute institution's average utility."""

    def sample_exogenous_noise(self, num_samps):
        pass

    def compute_output(self, exogenous_noise, utilities):  # pylint: disable=arguments-differ
        """Compute avg instit. utility.

        Assumes utilities is a batch of individual utilities."""
        return torch.mean(utilities)

@gin.configurable
class ThresholdScore(StructuralEqn):
    """Compute min(X, score_threshold)."""

    def __init__(self, score_threshold=gin.REQUIRED):
        self.score_threshold = float(score_threshold)

    def sample_exogenous_noise(self, num_samps):
        pass

    def compute_output(self, exogenous_noise, X):  # pylint: disable=arguments-differ
        output = torch.clamp(X, self.score_threshold)
        return output
