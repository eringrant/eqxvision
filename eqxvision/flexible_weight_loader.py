"""
Flexible weight loading utility for eqxvision models.
Uses name-based parameter mapping instead of positional alignment for better robustness.
"""

import logging
import os
import re
import sys
import warnings
from collections import defaultdict
from typing import Optional

import equinox as eqx
import jax.numpy as jnp


try:
    import torch
except ImportError:
    warnings.warn("PyTorch is required for loading pre-trained weights.")

_TEMP_DIR = "/tmp/.eqx"


def get_nested_attr(obj, path: str):
    """Get nested attribute using dot notation and array indexing."""
    parts = path.split(".")
    current = obj

    for part in parts:
        if "[" in part and "]" in part:
            attr_name, indices = part.split("[", 1)
            indices = indices.rstrip("]")
            current = getattr(current, attr_name)
            for idx in indices.split("]["):
                current = current[int(idx)]
        else:
            current = getattr(current, part)

    return current


def _normalize_pytorch_key(key: str) -> str:
    """Normalize PyTorch-specific naming patterns to numeric indices.

    Handles patterns like:
    - RegNet: ``block1.block1-0`` -> ``0.0``
    - RegNet: ``f.a`` -> ``f.0``, ``f.b`` -> ``f.1``, ``f.se`` -> ``f.2``, ``f.c`` -> ``f.3``
    - DenseNet: ``denseblock1.denselayer1`` -> ``0.0``
    """
    # RegNet: blockN.blockN-M -> (N-1).M
    key = re.sub(
        r"block(\d+)\.block\d+-(\d+)",
        lambda m: f"{int(m.group(1))-1}.{m.group(2)}",
        key,
    )
    # RegNet BottleneckTransform: f.a/b/se/c -> f.0/1/2/3
    _regnet_f_map = {"a": "0", "b": "1", "se": "2", "c": "3"}
    key = re.sub(
        r"\.f\.(a|b|se|c)\.", lambda m: f".f.{_regnet_f_map[m.group(1)]}.", key
    )
    # DenseNet: normalize features.X named children to indices
    # PT order: conv0, norm0, [relu0], [pool0], denseblock1, transition1,
    #           denseblock2, transition2, denseblock3, transition3, denseblock4, norm5
    # Eqx order: [0:Conv, 1:BN, 2:Lambda, 3:MaxPool, 4:DB, 5:Trans,
    #             6:DB, 7:Trans, 8:DB, 9:Trans, 10:DB, 11:BN, 12:Lambda, 13:AvgPool]
    _densenet_features_map = {
        "conv0": "0",
        "norm0": "1",
        "denseblock1": "4",
        "transition1": "5",
        "denseblock2": "6",
        "transition2": "7",
        "denseblock3": "8",
        "transition3": "9",
        "denseblock4": "10",
        "norm5": "11",
    }
    key = re.sub(
        r"features\.(conv0|norm0|norm5|denseblock\d+|transition\d+)\b",
        lambda m: f"features.{_densenet_features_map.get(m.group(1), m.group(1))}",
        key,
    )
    # DenseNet: denselayerM -> (M-1) within dense blocks
    key = re.sub(r"denselayer(\d+)", lambda m: str(int(m.group(1)) - 1), key)
    # DenseNet: norm.1 -> norm1, conv.1 -> conv1, etc (dotted naming quirk)
    # Only apply within dense block layers (after denselayer -> numeric conversion)
    # These look like: features.N.M.norm.1 or features.N.M.conv.1
    key = re.sub(r"(features\.\d+\.\d+\.(?:norm|conv))\.(\d+)", r"\1\2", key)
    # DenseNet transition: PT uses named attrs (norm, conv), eqx uses Sequential
    # [0:BN, 1:Lambda(relu), 2:Conv, 3:AvgPool]
    # Must apply AFTER features.transitionN -> features.N mapping
    _densenet_trans_indices = {"5", "7", "9"}
    for idx in _densenet_trans_indices:
        key = re.sub(rf"features\.{idx}\.norm(?!\.\d)", f"features.{idx}.layers.0", key)
        key = re.sub(rf"features\.{idx}\.conv(?!\.\d)", f"features.{idx}.layers.2", key)
    return key


def _convert_pytorch_key_to_equinox(key: str) -> Optional[str]:
    """Convert a PyTorch state dict key to an Equinox path.

    Numeric path components become ``layers[N]`` indexing.
    BatchNorm running stats are mapped to ``ema_state_index.init[0|1]``.
    """
    key = _normalize_pytorch_key(key)
    parts = key.split(".")

    converted = []
    for part in parts:
        if part == "running_mean":
            converted.append("ema_state_index.init[0]")
        elif part == "running_var":
            converted.append("ema_state_index.init[1]")
        elif part == "num_batches_tracked":
            return None
        elif part.isdigit():
            if converted:
                converted[-1] = converted[-1] + f".layers[{part}]"
            else:
                converted.append(f"layers[{part}]")
        else:
            converted.append(part)

    return ".".join(converted)


def _build_sequential_mapping(model, parent_path: str, attr_suffix: str):
    """For a Sequential at parent_path, return a list of eqxvision layer indices
    that have the given attribute suffix (e.g. 'weight'), in order."""
    try:
        parent = get_nested_attr(model, parent_path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return []

    if not hasattr(parent, "layers"):
        return []

    first_attr = attr_suffix.split(".")[0]
    result = []
    for i, layer in enumerate(parent.layers):
        try:
            get_nested_attr(layer, attr_suffix)
            result.append(i)
        except (AttributeError, IndexError, KeyError, TypeError):
            # Also try just the first attr component
            if hasattr(layer, first_attr):
                result.append(i)
    return result


def _try_wrapper_insertions(eqx_path: str, try_path_fn) -> Optional[str]:
    """Try inserting '.model' and '.layer' at various positions in the path.

    Handles IntermediateLayerGetter (.model) and IntermediateWrapper (.layer).
    """
    # Split path into segments (handling layers[N] as part of preceding segment)
    raw_parts = eqx_path.split(".")
    # Try inserting .model and .layer at each dot boundary
    _inserts = [".model", ".layer"]
    # Try single insertions first, then pairs
    for insert in _inserts:
        for i in range(1, len(raw_parts)):
            candidate = ".".join(raw_parts[:i]) + insert + "." + ".".join(raw_parts[i:])
            if try_path_fn(candidate) is not None:
                return candidate
    # Try .model after first component + .layer somewhere after
    first_dot = eqx_path.find(".")
    if first_dot > 0:
        model_path = eqx_path[:first_dot] + ".model" + eqx_path[first_dot:]
        model_parts = model_path.split(".")
        for i in range(2, len(model_parts)):
            candidate = (
                ".".join(model_parts[:i]) + ".layer." + ".".join(model_parts[i:])
            )
            if try_path_fn(candidate) is not None:
                return candidate
    return None


def flexible_load_torch_weights(
    model: eqx.Module, torch_weights: str = None, verbose: bool = True
) -> eqx.Module:
    """Load PyTorch weights into Equinox model using flexible name-based mapping.

    **Arguments:**

    - `model`: An `eqx.Module` for which parameters will be loaded
    - `torch_weights`: A string pointing to PyTorch weights on disk or download URL
    - `verbose`: Whether to print loading progress. Defaults to `True`

    **Returns:**
        The model with weights loaded from the PyTorch checkpoint.
    """
    if "torch" not in sys.modules:
        raise RuntimeError(
            "Torch package not found! Weight loading requires the torch package."
        )

    if torch_weights is None:
        raise ValueError("torch_weights parameter cannot be empty!")

    # Download weights if needed
    if not os.path.exists(torch_weights):
        global _TEMP_DIR
        filepath = os.path.join(_TEMP_DIR, os.path.basename(torch_weights))
        if os.path.exists(filepath):
            if verbose:
                logging.info(f"Using cached file at {filepath}")
        else:
            os.makedirs(_TEMP_DIR, exist_ok=True)
            if verbose:
                print(f"Downloading weights from {torch_weights}")
            torch.hub.download_url_to_file(torch_weights, filepath)
    else:
        filepath = torch_weights

    pytorch_state_dict = torch.load(filepath, map_location="cpu", weights_only=False)

    if verbose:
        print("=== Flexible Weight Loading ===")

    successful_loads = 0
    failed_loads = []
    skipped = 0

    # Counter for sequential fallback: tracks how many times we've seen
    # a particular (parent_path, attr_suffix) combo to determine ordinal.
    seq_ordinal_counter = defaultdict(int)
    # Cache for sequential mappings
    seq_mapping_cache = {}

    for pytorch_name, pytorch_weight in pytorch_state_dict.items():
        if "num_batches_tracked" in pytorch_name:
            skipped += 1
            continue

        eqx_path = _convert_pytorch_key_to_equinox(pytorch_name)
        if eqx_path is None:
            skipped += 1
            continue

        pytorch_array = jnp.asarray(pytorch_weight.detach().numpy())

        def _try_path(path):
            """Try to resolve path and check shape compatibility."""
            try:
                param = get_nested_attr(model, path)
            except (AttributeError, IndexError, KeyError, TypeError):
                return None
            if not hasattr(param, "shape"):
                return None
            if param.shape == pytorch_array.shape or param.size == pytorch_array.size:
                return param
            return None

        # Try the converted path directly
        resolved = False
        current_param = _try_path(eqx_path)
        if current_param is not None:
            resolved = True

        # Fallback: try inserting '.model' and/or '.layer' to handle
        # IntermediateLayerGetter wrapping (adds .model) and
        # IntermediateWrapper wrapping (adds .layer)
        if not resolved:
            candidate = _try_wrapper_insertions(eqx_path, _try_path)
            if candidate is not None:
                eqx_path = candidate
                current_param = _try_path(candidate)
                resolved = True

        # Fallback: try raw pytorch key
        if not resolved:
            current_param = _try_path(pytorch_name)
            if current_param is not None:
                eqx_path = pytorch_name
                resolved = True

        # Track ordinal for any key that involves layers[N] (whether resolved or not),
        # so the sequential fallback counter stays in sync.
        seq_match = re.search(r"^(.+)\.layers\[(\d+)\]\.(.+)$", eqx_path)
        if seq_match and resolved:
            cache_key = (seq_match.group(1), seq_match.group(3))
            seq_ordinal_counter[cache_key] += 1

        # Fallback: sequential index remapping
        if not resolved:
            m = re.search(r"^(.+)\.layers\[(\d+)\]\.(.+)$", eqx_path)
            if m:
                parent_path = m.group(1)
                attr_suffix = m.group(3)
                cache_key = (parent_path, attr_suffix)

                if cache_key not in seq_mapping_cache:
                    seq_mapping_cache[cache_key] = _build_sequential_mapping(
                        model, parent_path, attr_suffix
                    )

                eqx_indices = seq_mapping_cache[cache_key]
                ordinal = seq_ordinal_counter[cache_key]
                seq_ordinal_counter[cache_key] += 1

                if ordinal < len(eqx_indices):
                    eqx_idx = eqx_indices[ordinal]
                    candidate = f"{parent_path}.layers[{eqx_idx}].{attr_suffix}"
                    current_param = _try_path(candidate)
                    if current_param is not None:
                        eqx_path = candidate
                        resolved = True

        # Fallback: try wrapper insertions on the unresolved path
        if not resolved:
            candidate = _try_wrapper_insertions(eqx_path, _try_path)
            if candidate is not None:
                eqx_path = candidate
                current_param = _try_path(candidate)
                resolved = True

        if not resolved:
            failed_loads.append((pytorch_name, eqx_path, "no matching path"))
            continue

        # Handle shape mismatch with same element count (e.g. bias (C,) -> (C,1,1))
        if pytorch_array.shape != current_param.shape:
            if pytorch_array.size == current_param.size:
                pytorch_array = jnp.reshape(pytorch_array, current_param.shape)
            else:
                if verbose:
                    print(
                        f"Shape mismatch: {pytorch_name} {pytorch_array.shape} vs {eqx_path} {current_param.shape}"
                    )
                failed_loads.append((pytorch_name, eqx_path, "shape_mismatch"))
                continue

        where_fn = lambda m, p=eqx_path: get_nested_attr(m, p)
        model = eqx.tree_at(where_fn, model, pytorch_array)

        if verbose:
            print(f"  {pytorch_name} -> {eqx_path} {pytorch_array.shape}")
        successful_loads += 1

    if verbose:
        print("\n=== Loading Results ===")
        print(f"Parameters loaded: {successful_loads}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {len(failed_loads)}")

        if failed_loads and len(failed_loads) <= 20:
            print("\nFailed loads:")
            for pt_name, eqx_path, reason in failed_loads:
                print(f"  {pt_name} -> {eqx_path}: {reason}")

    return model
