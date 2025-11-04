"""
Normalization utilities for eqxvision models.

Provides two main utilities:
1. StatelessBatchNorm - for inference without state tracking
2. replace_norm - for converting BatchNorm to StatelessBatchNorm or GroupNorm
"""

from typing import Literal, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array


class StatelessBatchNorm(eqx.Module):
    """Stateless BatchNorm for inference - uses fixed running statistics.
    
    No state tracking needed, no vmap axis_name required.
    """
    
    weight: jnp.ndarray
    bias: jnp.ndarray  
    running_mean: jnp.ndarray
    running_var: jnp.ndarray
    eps: float
    
    def __init__(self, bn_layer: eqx.nn.BatchNorm):
        """Convert a BatchNorm layer to stateless form."""
        self.weight = bn_layer.weight if bn_layer.weight is not None else jnp.ones(bn_layer.input_size)
        self.bias = bn_layer.bias if bn_layer.bias is not None else jnp.zeros(bn_layer.input_size)
        
        # Extract running statistics
        if hasattr(bn_layer, 'ema_state_index') and bn_layer.ema_state_index is not None:
            self.running_mean, self.running_var = bn_layer.ema_state_index.init
        else:
            self.running_mean = jnp.zeros(bn_layer.input_size)
            self.running_var = jnp.ones(bn_layer.input_size)
            
        self.eps = bn_layer.eps

    def __call__(self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None) -> Array:
        """Apply batch normalization using fixed statistics."""
        # (x - mean) / sqrt(var + eps) * weight + bias
        if x.ndim == 3:  # Conv: (C, H, W)
            normalized = (x - self.running_mean.reshape(-1, 1, 1)) / jnp.sqrt(self.running_var.reshape(-1, 1, 1) + self.eps)
            return normalized * self.weight.reshape(-1, 1, 1) + self.bias.reshape(-1, 1, 1)
        elif x.ndim == 1:  # FC: (C,)
            normalized = (x - self.running_mean) / jnp.sqrt(self.running_var + self.eps)
            return normalized * self.weight + self.bias
        else:
            raise ValueError(f"Expected 1D or 3D input, got shape {x.shape}")


def replace_norm(
    model: eqx.Module,
    target: Literal["stateless", "groupnorm"] = "stateless"
) -> eqx.Module:
    """Replace BatchNorm layers in a model.
    
    Args:
        model: Model containing BatchNorm layers
        target: "stateless" for StatelessBatchNorm (inference only)
                "groupnorm" for GroupNorm (training/finetuning with small batches)
    
    Returns:
        Model with BatchNorm replaced
    
    Examples:
        # For inference (no state tracking)
        model = replace_norm(model, target="stateless")
        
        # For imitation learning / RL (GroupNorm, num_groups = channels // 16)
        model = replace_norm(model, target="groupnorm")
    """
    def replace_fn(module):
        if isinstance(module, eqx.nn.BatchNorm):
            if target == "stateless":
                return StatelessBatchNorm(module)
            elif target == "groupnorm":
                num_channels = module.input_size
                num_groups = max(1, num_channels // 16)
                return eqx.nn.GroupNorm(groups=num_groups, channels=num_channels)
            else:
                raise ValueError(f"target must be 'stateless' or 'groupnorm', got {target}")
        return module
    
    return jtu.tree_map(replace_fn, model, is_leaf=lambda x: isinstance(x, eqx.nn.BatchNorm))
