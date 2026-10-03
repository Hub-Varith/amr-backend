import numpy as np
import torch
from scipy.stats import norm

from genome2mic.models.censored_normal_loss import CensoredNormalLoss


def closed_form(mu, sigma, lower, upper):
    return -np.log(norm.cdf((upper - mu) / sigma) - norm.cdf((lower - mu) / sigma))


def test_matches_scipy_for_interval_left_and_right_censoring():
    mu = torch.tensor([[1.0, 1.0, 1.0]])
    log_sigma = torch.log(torch.tensor([0.8, 0.8, 0.8]))
    lower = torch.tensor([[0.0, -np.inf, 3.0]])
    upper = torch.tensor([[2.0, 0.5, np.inf]])
    mask = torch.ones_like(mu)
    loss = CensoredNormalLoss()(mu, log_sigma, lower, upper, mask).item()
    expected = np.mean([closed_form(1.0, 0.8, 0.0, 2.0), closed_form(1.0, 0.8, -np.inf, 0.5), closed_form(1.0, 0.8, 3.0, np.inf)])
    assert abs(loss - expected) < 1e-5


def test_masked_entries_do_not_change_the_loss():
    mu = torch.tensor([[0.0, 50.0]])
    log_sigma = torch.zeros(2)
    lower = torch.tensor([[-1.0, -1.0]])
    upper = torch.tensor([[1.0, 1.0]])
    masked = CensoredNormalLoss()(mu, log_sigma, lower, upper, torch.tensor([[1.0, 0.0]])).item()
    alone = CensoredNormalLoss()(mu[:, :1], log_sigma[:1], lower[:, :1], upper[:, :1], torch.ones(1, 1)).item()
    assert abs(masked - alone) < 1e-6


def test_far_away_interval_stays_finite_with_gradient():
    mu = torch.tensor([[-20.0, 20.0]], requires_grad=True)
    log_sigma = torch.zeros(2)
    lower = torch.tensor([[5.0, -np.inf]])
    upper = torch.tensor([[6.0, -5.0]])
    loss = CensoredNormalLoss()(mu, log_sigma, lower, upper, torch.ones(1, 2))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(mu.grad).all()
    assert mu.grad[0, 0] < 0 and mu.grad[0, 1] > 0


def test_drug_balanced_weights_each_drug_equally():
    mu = torch.zeros(4, 2)
    log_sigma = torch.zeros(2)
    lower = torch.full((4, 2), -1.0)
    upper = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    upper[:, 1] = 0.0
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    plain = CensoredNormalLoss(drug_balanced=False)(mu, log_sigma, lower, upper, mask).item()
    balanced = CensoredNormalLoss(drug_balanced=True)(mu, log_sigma, lower, upper, mask).item()
    drug_a = closed_form(0.0, 1.0, -1.0, 1.0)
    drug_b = closed_form(0.0, 1.0, -1.0, 0.0)
    assert abs(plain - (4 * drug_a + drug_b) / 5) < 1e-5
    assert abs(balanced - (drug_a + drug_b) / 2) < 1e-5


def test_nan_bounds_under_the_mask_are_ignored():
    mu = torch.tensor([[0.0, 0.0]], requires_grad=True)
    log_sigma = torch.zeros(2)
    lower = torch.tensor([[-1.0, np.nan]])
    upper = torch.tensor([[1.0, np.nan]])
    loss = CensoredNormalLoss()(mu, log_sigma, lower, upper, torch.tensor([[1.0, 0.0]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(mu.grad).all()
    assert abs(loss.item() - closed_form(0.0, 1.0, -1.0, 1.0)) < 1e-6


def test_infinite_bounds_give_finite_sigma_gradient():
    mu = torch.zeros(1, 2)
    log_sigma = torch.zeros(2, requires_grad=True)
    lower = torch.tensor([[-np.inf, 2.0]])
    upper = torch.tensor([[-1.0, np.inf]])
    loss = CensoredNormalLoss()(mu, log_sigma, lower, upper, torch.ones(1, 2))
    loss.backward()
    assert torch.isfinite(log_sigma.grad).all()
