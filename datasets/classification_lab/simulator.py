"""Simulator for Lab 7 (proxy losses, logistic regression, LDA and ROC curves).

This file defines "practice worlds": binary classification problems with two
features, in which each class is drawn from a known distribution, so that we
know the true probability p(x) = P(y = 1 | x). Because we know it, we can
compute the true risk of any fitted model, draw the true (Bayes) decision
boundary, and build the Bayes classifier itself.

There are two scenarios:

* "shared":      x | y = k ~ N(mu_k, Sigma), the same covariance in both
                 classes. This is exactly the LDA model.
* "far_cluster": like "shared", except that 10% of the class 1 observations
                 come from a small, far-away cluster. These observations are
                 easy to classify, but the classes are no longer Gaussian.

The class proportion P(y = 1) can be changed with `prior`, without changing
how the features are distributed within each class. Small values of `prior`
give a rare class 1.

Usage:

    from simulator import PracticeWorld

    world = PracticeWorld("shared")       # or PracticeWorld("shared", prior=0.05)
    X, y = world.simulate(200, seed=1)    # a training set of N = 200
    model = make_model().fit(X, y)        # any model with .predict_proba
    world.true_risk(model)                # expected log loss on new data
    world.true_risk(model, loss="error")  # misclassification rate on new data
    world.true_prob(X)                    # the true p(x) for each row of X
    world.plot_true_boundary(ax)          # draw p(x) = 0.5
"""

import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal

SCENARIOS = ("shared", "far_cluster")
FAR_WEIGHT = 0.1    # fraction of class 1 that comes from the far cluster


class PracticeWorld:
    """
    A binary classification problem with two features and Gaussian (or almost Gaussian) classes.

    Parameters
    ----------
    scenario : {"shared", "far_cluster"}
        Which practice world to use (see the module docstring).
    prior : float, optional
        P(y = 1). Default is 0.5.
    """

    def __init__(self, scenario, prior=0.5):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.scenario, self.prior = scenario, prior

        self.mu0 = np.array([0.0, 0.0])
        self.mu1 = np.array([1.5, -0.5])
        self.sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
        self.mu_far = np.array([8.0, -8.0])
        self.sigma_far = 0.25 * np.eye(2)

        # A large, fixed sample of new feature vectors used to compute the true risk.
        # Its seed is a list, so it never coincides with simulate(n, seed=<integer>).
        self._X_eval = self.simulate(20_000, seed=[154, 2026])[0]
        self._p_eval = self.true_prob(self._X_eval)

    def simulate(self, n, seed=None):
        """
        Draw n IID observations (X, y).

        Parameters
        ----------
        n : int
            Number of observations.
        seed : int or sequence of int, optional
            Seed of the random number generator. Default is None (random).

        Returns
        -------
        X : ndarray of shape (n, 2)
            Features.
        y : ndarray of shape (n,)
            Binary labels.
        """
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, self.prior, size=n)
        X = np.empty((n, 2))
        n1 = y.sum()
        X[y == 0] = rng.multivariate_normal(self.mu0, self.sigma, size=n - n1)
        X[y == 1] = rng.multivariate_normal(self.mu1, self.sigma, size=n1)
        if self.scenario == "far_cluster":
            far = (y == 1) & (rng.random(n) < FAR_WEIGHT)
            X[far] = rng.multivariate_normal(self.mu_far, self.sigma_far, size=far.sum())
        return X, y

    def true_prob(self, X):
        """
        True probability P(y = 1 | x) for each row of X, from Bayes' rule.

        Parameters
        ----------
        X : ndarray of shape (n, 2)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n,)
            The probability that y = 1 given each row of `X`.
        """
        # Work with log densities, so that points far from both classes do not give 0 / 0.
        log_dens0 = multivariate_normal(self.mu0, self.sigma).logpdf(X)
        log_dens1 = multivariate_normal(self.mu1, self.sigma).logpdf(X)
        if self.scenario == "far_cluster":
            log_dens_far = multivariate_normal(self.mu_far, self.sigma_far).logpdf(X)
            log_dens1 = np.logaddexp(np.log(1 - FAR_WEIGHT) + log_dens1,
                                     np.log(FAR_WEIGHT) + log_dens_far)
        log_odds = np.log(self.prior) + log_dens1 - np.log(1 - self.prior) - log_dens0
        return expit(log_odds)

    def true_risk(self, model, loss="log_loss", return_se=False):
        """
        Expected loss of a fitted model on a new observation.

        Averages over 20,000 fixed new feature vectors, with each random label
        replaced by its true probability to remove label noise.

        Parameters
        ----------
        model : object
            Fitted model with a ``predict_proba`` method.
        loss : {"log_loss", "error"}, optional
            "log_loss" (default) scores the predicted probabilities q(x):
            ``E[-p(x) log q(x) - (1 - p(x)) log(1 - q(x))]``. "error" scores
            the classifier that predicts 1 when q(x) > 0.5: the probability
            that it misclassifies a new observation.
        return_se : bool, optional
            If True, also return the Monte Carlo standard error. Default False.

        Returns
        -------
        risk : float
            The true risk of `model`.
        se : float
            Monte Carlo standard error of `risk`. Only returned if `return_se`.
        """
        q = model.predict_proba(self._X_eval)[:, 1]
        losses = _losses(self._p_eval, q, loss)
        risk = losses.mean()
        if return_se:
            return risk, losses.std(ddof=1) / np.sqrt(len(losses))
        return risk

    def bayes_risk(self, loss="log_loss"):
        """
        Smallest possible risk, achieved by predicting the true probability.

        Parameters
        ----------
        loss : {"log_loss", "error"}, optional
            See `true_risk`. Default is "log_loss".

        Returns
        -------
        float
            The risk of the model that predicts `true_prob(x)` exactly.
        """
        return _losses(self._p_eval, self._p_eval, loss).mean()

    def plot_true_boundary(self, ax, level=0.5, **kwargs):
        """
        Draw the curve {x : p(x) = level} on a plot of the two features.

        Parameters
        ----------
        ax : matplotlib Axes
            Axes with the first feature on the x axis and the second on the y axis.
            Draw your data first, so that the axis limits are already set.
        level : float, optional
            Which level of p(x) to draw. Default is 0.5, the Bayes decision boundary.
        **kwargs
            Passed to ``ax.contour``, e.g. ``colors="k"``.
        """
        # Evaluate on a grid that extends well beyond the current limits, so the
        # boundary still reaches the edges if the limits grow later, then restore them.
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x0, x1 = np.meshgrid(np.linspace(xlim[0] - dx, xlim[1] + dx, 600),
                             np.linspace(ylim[0] - dy, ylim[1] + dy, 600))
        prob = self.true_prob(np.column_stack([x0.ravel(), x1.ravel()])).reshape(x0.shape)
        kwargs = {"colors": "k", "linestyles": "--", **kwargs}
        ax.contour(x0, x1, prob, levels=[level], **kwargs)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def _losses(p, q, loss):
    """
    Per-observation loss of predictions q, averaged over the random label.

    Parameters
    ----------
    p : ndarray of shape (m,)
        True probabilities P(y = 1 | x).
    q : ndarray of shape (m,)
        Predicted probabilities.
    loss : {"log_loss", "error"}
        Which loss to use (see `PracticeWorld.true_risk`).

    Returns
    -------
    ndarray of shape (m,)
        The expected loss for each observation.
    """
    if loss == "log_loss":
        q = np.clip(q, 1e-15, 1 - 1e-15)
        return -p * np.log(q) - (1 - p) * np.log(1 - q)
    if loss == "error":
        return np.where(q > 0.5, 1 - p, p)
    raise ValueError('loss must be "log_loss" or "error"')
