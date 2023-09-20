"""Loss functions for probabilistic regression models.

This module contains the following classes:
    CRPSLoss: Continuous Ranked Probability Score (CRPS) loss function.
    EnsembleVarRegLoss: Ensemble variance regularization loss function.
    MaskedLoss: Masked loss function.
    
"""
# Standard library
from typing import Any

# Third-party
import torch
from torch import nn
from torch.distributions import Normal

from gwen.loggers_configs import setup_logger

logger = setup_logger()


class CRPSLoss(nn.Module):
    """Continuous Ranked Probability Score (CRPS) loss function.

    This class implements the CRPS loss function, which is used to eval the
    performance of probabilistic regression models.

    Args:
        nn.Module: PyTorch module.

    Returns:
        crps_loss: CRPS loss for each sample in the batch.

    """

    def __init__(self) -> None:
        """Initialize the CRPS loss function."""
        super().__init__()

    def forward(self, outputs: Any, target: Any, dim: int = 0) -> Any:
        """Calculate the CRPS loss for each sample in the batch.

        This method calculates the CRPS loss for each sample in the batch using the
        predicted values and target values.

        Args:
            outputs: Predicted values.
            target: Target values.
            dim: Dimension over which to calculate the mean and standard deviation.

        Returns:
            crps_loss: CRPS loss for each sample in the batch.

        """
        # Calculate the mean and standard deviation of the predicted distribution
        try:
            mu = torch.mean(outputs, dim=dim)  # Mean over ensemble members
            # Stddev over ensemble members
            sigma = torch.std(outputs, dim=dim) + 1e-6

            # Create a normal distribution with the predicted mean and standard
            # deviation
            dist = Normal(mu, sigma)

            # Calculate the CRPS loss for each sample in the batch Mean over ensemble
            # members and spatial locations
            crps_loss = torch.mean(
                (dist.cdf(target) - 0.5) ** 2, dim=[1, 2, 3])

            return crps_loss
        except Exception as e:
            logger.exception("Error calculating CRPS loss: %s", e)
            raise


class EnsembleVarRegLoss(nn.Module):
    """Ensemble variance regularization loss function.

    This class implements the ensemble variance regularization loss function, which is
    used to improve the performance of probabilistic regression models.

    Args:
        alpha: Regularization strength.

    Returns:
        l1_loss + regularization_loss: Loss for each sample in the batch.

    """

    def __init__(self, alpha: float = 0.1) -> None:
        """Initialize the ensemble variance regularization loss function.

        Args:
            alpha: Regularization strength.

        """
        super().__init__()
        self.alpha = alpha  # Regularization strength

    def forward(self, outputs: Any, target: Any) -> Any:
        """Calculate the loss for each sample using the specified loss function.

        This method calculates the loss for each sample in the batch using the specified
        loss function.

        Args:
            outputs: Predicted values.
            target: Target values.

        Returns:
            l1_loss + regularization_loss: Loss for each sample in the batch.

        """
        try:
            l1_loss = torch.mean(torch.abs(outputs - target))
            ensemble_variance = torch.var(outputs, dim=1)
            regularization_loss = -self.alpha * torch.mean(ensemble_variance)
            return l1_loss + regularization_loss
        except Exception as e:
            logger.exception(
                "Error calculating ensemble variance regularization loss: %s", e
            )
            raise


class MaskedLoss(nn.Module):
    """Masked loss function.

    This class implements the masked loss function, which is used to calculate the loss
    for each sample in the batch while ignoring certain cells.

    Args:
        loss_fn: Loss function to use.

    Returns:
        mean_loss: Mean loss over all unmasked cells.

    """

    def __init__(self, loss_fn: nn.Module) -> None:
        """Initialize the masked loss function.

        Args:
            loss_fn: Loss function to use.

        """
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, outputs: Any, target: Any, mask: Any) -> Any:
        """Calculate the loss for each sample using the specified loss function.

        This method calculates the loss for each sample in the batch using the specified
        loss function, while ignoring certain cells.

        Args:
            outputs: Predicted values.
            target: Target values.
            mask: Mask for cells where the values stay constant over all observed times.

        Returns:
            mean_loss: Mean loss over all unmasked cells.

        """
        try:
            # Calculate the loss for each sample in the batch using the specified loss
            # function
            loss = self.loss_fn(outputs, target)

            # Mask the loss for cells where the values stay constant over all observed
            # times
            masked_loss = loss * mask

            # Calculate the mean loss over all unmasked cells
            mean_loss = torch.sum(masked_loss) / torch.sum(mask)

            return mean_loss
        except Exception as e:
            logger.exception("Error calculating masked loss: %s", e)
            raise
