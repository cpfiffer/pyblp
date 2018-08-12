"""Standard statistical routines."""

import warnings

import numpy as np
import scipy.linalg

from .. import exceptions


class IV(object):
    """Simple model for generalized instrumental variables estimation."""

    def __init__(self, X, Z, W):
        """Store data and pre-compute covariances."""
        self.X = X
        self.Z = Z
        self.W = W

        # attempt to pre-compute covariances
        covariances_inverse = (self.X.T @ self.Z) @ self.W @ (self.Z.T @ self.X)
        self.covariances, replacement = approximately_invert(covariances_inverse)

        # store any errors
        self.errors = []
        if replacement:
            self.errors.append(exceptions.LinearParameterCovariancesInversionError(covariances_inverse, replacement))

    def estimate(self, y, compute_residuals=True):
        """Estimate parameters and optionally compute residuals."""
        parameters = self.covariances @ (self.X.T @ self.Z) @ self.W @ (self.Z.T @ y)
        if compute_residuals:
            return parameters, y - self.X @ parameters
        return parameters


def compute_gmm_se(u, Z, W, jacobian, se_type, clustering_ids):
    """Use an error term, instruments, a weighting matrix, and the Jacobian of the error term with respect to parameters
    to estimate GMM standard errors.
    """
    errors = []

    # compute the Jacobian of the sample moments with respect to all parameters
    G = Z.T @ jacobian

    # attempt to compute the covariance matrix
    covariances_inverse = G.T @ W @ G
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the robust covariance matrix and extract standard errors
    with np.errstate(invalid='ignore'):
        if se_type != 'unadjusted':
            g = u * Z
            S = compute_gmm_moment_covariances(g, se_type, clustering_ids)
            covariances = covariances @ G.T @ W @ S @ W @ G @ covariances
        se = np.sqrt(np.c_[covariances.diagonal()])

    # handle null values
    if np.isnan(se).any():
        errors.append(exceptions.InvalidParameterCovariancesError())
    return se, errors


def compute_2sls_weights(Z):
    """Use instruments to compute a 2SLS weighting matrix."""
    errors = []

    # attempt to compute the weighting matrix
    covariances = Z.T @ Z
    W, replacement = approximately_invert(covariances)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(covariances, replacement))
    return W, errors


def compute_gmm_weights(u, Z, center_moments, se_type, clustering_ids):
    """Use an error term and instruments to compute a GMM weighting matrix."""
    errors = []

    # compute and center the sample moments
    g = u * Z
    if center_moments:
        g -= g.mean(axis=0)

    # attempt to compute the weighting matrix
    covariances = compute_gmm_moment_covariances(g, se_type, clustering_ids)
    W, replacement = approximately_invert(covariances)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(covariances, replacement))

    # handle null values
    if np.isnan(W).any():
        errors.append(exceptions.InvalidMomentCovariancesError())
    return W, errors


def compute_gmm_moment_covariances(g, se_type, clustering_ids):
    """Compute covariances between moment conditions."""
    if se_type == 'clustered' and clustering_ids.shape[1] > 0:
        return sum(g[clustering_ids.flat == c].T @ g[clustering_ids.flat == c] for c in np.unique(clustering_ids))
    return g.T @ g


def precisely_solve(a, b):
    """Attempt to precisely solve a system of equations."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            solved = scipy.linalg.solve(a, b) if b.size > 0 else b
            successful = True
    except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        solved = np.full_like(b, np.nan)
        successful = False
    return solved, successful


def precisely_invert(x):
    """Attempt to precisely invert a matrix."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            inverted = scipy.linalg.inv(x) if x.size > 0 else x
            successful = True
    except (ValueError, scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        inverted = np.full_like(x, np.nan)
        successful = False
    return inverted, successful


def approximately_solve(a, b):
    """Attempt to solve a system of equations with decreasingly precise replacements for the inverse."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            solved = scipy.linalg.solve(a, b) if b.size > 0 else b
            replacement = None
    except:
        inverse, replacement = approximately_invert(a)
        solved = inverse @ b
    return solved, replacement


def approximately_invert(x):
    """Attempt to invert a matrix with decreasingly precise replacements for the inverse."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            inverted = scipy.linalg.inv(x) if x.size > 0 else x
            replacement = None
    except ValueError:
        inverted = np.full_like(x, np.nan)
        replacement = "null values"
    except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        try:
            inverted = scipy.linalg.pinv(x)
            replacement = "its Moore-Penrose pseudo inverse"
        except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
            inverted = np.diag(1 / np.diag(x))
            replacement = "inverted diagonal terms because the Moore-Penrose pseudo inverse could not be computed"
    return inverted, replacement
