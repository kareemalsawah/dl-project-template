"""
Helpers function for losses
"""
import torch
import numpy as np


def gaussian_nll(
    mu: torch.Tensor, log_var: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian negative log likelihood
    From theano/models/__init__.py

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the Gaussian
    log_var : torch.Tensor
        Log variance of the Gaussian
    x: torch.Tensor
        Observation

    Returns
    -------
    torch.Tensor
        Negative log likelihood of the observation under the Gaussian
    """
    return 0.5 * (np.log(2 * np.pi) + log_var + (x - mu) ** 2 / torch.exp(log_var))


def gaussian_kl(
    mu_q: torch.Tensor,
    log_cov_q: torch.Tensor,
    mu_prior: torch.Tensor,
    log_cov_prior: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence between two Gaussians
    From theano/models/__init__.py

    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of the first Gaussian
    log_cov_q : torch.Tensor
        Log covariance of the first Gaussian
    mu_prior : torch.Tensor
        Mean of the second Gaussian
    log_cov_prior : torch.Tensor
        Log covariance of the second Gaussian

    Returns
    -------
    torch.Tensor
        KL divergence between the two Gaussians
    """
    diff_mu = mu_prior - mu_q
    cov_q = torch.exp(log_cov_q)
    cov_prior = torch.exp(log_cov_prior)
    kl_div = (
        log_cov_prior - log_cov_q - 1.0 + cov_q / cov_prior + diff_mu**2 / cov_prior
    )
    kl_div *= 0.5
    return kl_div


def entropy_upper_bound(samples: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy upper bound of a set of samples

    Parameters
    ----------
    samples : torch.Tensor
        Samples to compute the entropy upper bound of
        shape = (n_samples, batch_size, n_time_steps, latent_dim)

    Returns
    -------
    torch.Tensor
        Entropy upper bound of the samples
    """
    standard_deviation = torch.std(samples, dim=0)
    term1 = 0.5 * torch.log(standard_deviation.sum(dim=2))
    term2 = 0.5 * (1 + np.log(2 * np.pi)) * samples.shape[3]
    return term1 + term2
