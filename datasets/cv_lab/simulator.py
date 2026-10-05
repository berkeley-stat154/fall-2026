"""Simulator for Lab 5 (cross validation).

This file defines a "practice world": a binary classification problem where we
know the true probability P(y = 1 | x). Because we know it, we can compute the
true risk (expected log loss on new data) of any fitted model, and check how
well training error, validation error and cross validation error estimate it.

Usage:

    from simulator import simulate, true_risk, bayes_risk

    X, y = simulate(500, seed=1)        # a training set of N = 500 observations
    model = make_model().fit(X, y)       # any model with .predict_proba
    true_risk(model)                     # expected log loss of model on new data
"""

import numpy as np

P = 10    # number of features


def true_prob(X):
    """
    True probability P(y = 1 | x) for each row of X.

    Parameters
    ----------
    X : ndarray of shape (n, P)
        Feature matrix.

    Returns
    -------
    ndarray of shape (n,)
        The probability that y = 1 given each row of `X`.
    """
    logit = (0.2
             + 1.0 * X[:, 0]
             - 0.8 * (X[:, 1] ** 2 - 1)
             + 1.2 * np.sin(2 * X[:, 2])
             + 1.5 * (X[:, 3] > 0) * X[:, 4]
             - 1.0 * (np.abs(X[:, 5]) > 1))
    return 1 / (1 + np.exp(-logit))


def simulate(n, seed=None):
    """
    Draw n IID observations (X, y). Features 6-9 are pure noise.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int or sequence of int, optional
        Seed of the random number generator. Default is None (random).

    Returns
    -------
    X : ndarray of shape (n, P)
        Standard normal features.
    y : ndarray of shape (n,)
        Binary labels, drawn with probability `true_prob(X)`.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P))
    y = rng.binomial(1, true_prob(X))
    return X, y


def _expected_log_losses(p, q):
    """
    Per-observation log loss of predictions q, averaged over the random label.

    Parameters
    ----------
    p : ndarray of shape (m,)
        True probabilities P(y = 1 | x).
    q : ndarray of shape (m,)
        Predicted probabilities.

    Returns
    -------
    ndarray of shape (m,)
        ``-p log q - (1 - p) log(1 - q)`` for each observation.
    """
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return -p * np.log(q) - (1 - p) * np.log(1 - q)


# A large, fixed sample of new feature vectors used to compute the true risk.
# Its seed is a list, so it never coincides with simulate(n, seed=<integer>).
_X_EVAL = simulate(20_000, seed=[154, 2026])[0]
_P_EVAL = true_prob(_X_EVAL)


def true_risk(model, return_se=False):
    """
    Expected log loss of a fitted model on a new observation.

    Computes ``E_x[-p(x) log q(x) - (1 - p(x)) log(1 - q(x))]``, where p is
    the true probability and q is the model's prediction, by averaging over
    20,000 fixed new feature vectors.

    Parameters
    ----------
    model : object
        Fitted model with a ``predict_proba`` method.
    return_se : bool, optional
        If True, also return the Monte Carlo standard error of the average,
        i.e. the sd of the per-observation losses divided by the square root
        of 20,000. Default is False.

    Returns
    -------
    risk : float
        The true risk of `model`.
    se : float
        Monte Carlo standard error of `risk`. Only returned if `return_se`.
    """
    losses = _expected_log_losses(_P_EVAL, model.predict_proba(_X_EVAL)[:, 1])
    risk = losses.mean()
    if return_se:
        return risk, losses.std(ddof=1) / np.sqrt(len(losses))
    return risk


def bayes_risk():
    """
    Smallest possible risk, achieved by predicting the true probability.

    Returns
    -------
    float
        The risk of the model that predicts `true_prob(x)` exactly.
    """
    return _expected_log_losses(_P_EVAL, _P_EVAL).mean()
