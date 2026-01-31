"""Benchmark `eqxvision` against `torchvision`.

Tests that `eqxvision` models produce identical outputs and gradients to their
`torchvision` counterparts when loaded with the same pretrained weights.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import torch
import torchvision.models as tv_models

from eqxvision.models import (
    alexnet,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    googlenet,
    mobilenet_v2,
    mobilenet_v3_large,
    regnet_y_400mf,
    resnet18,
    resnet50,
    shufflenet_v2_x1_0,
    squeezenet1_0,
    vgg16,
    vgg16_bn,
)
from eqxvision.models.segmentation.deeplabv3 import deeplabv3
from eqxvision.models.segmentation.fcn import fcn
from eqxvision.models.segmentation.lraspp import lraspp_mobilenet_v3_large


pytestmark = [
    pytest.mark.slow,
    pytest.mark.benchmark,
]

# Reason string for models using the legacy weight loader.
_LEGACY_LOADER = "uses legacy `load_torch_weights` which is fragile"


@dataclass
class ModelSpec:
    name: str
    eqx_factory: Callable
    tv_factory: Callable
    tv_weights: Any
    is_segmentation: bool = False
    input_size: tuple = (3, 224, 224)
    eqx_kwargs: dict = field(default_factory=dict)
    tv_kwargs: dict = field(default_factory=dict)
    xfail: str | None = None  # If set, mark test as `xfail` with this reason


MODEL_SPECS = [
    # --- Models using `flexible_load_torch_weights` (expected to pass) ---
    ModelSpec(
        name="resnet18",
        eqx_factory=resnet18,
        tv_factory=tv_models.resnet18,
        tv_weights=tv_models.ResNet18_Weights.DEFAULT,
    ),
    ModelSpec(
        name="resnet50",
        eqx_factory=resnet50,
        tv_factory=tv_models.resnet50,
        tv_weights=tv_models.ResNet50_Weights.IMAGENET1K_V2,
    ),
    # --- Models using legacy `load_torch_weights` (expected to fail) ---
    ModelSpec(
        name="vgg16",
        eqx_factory=vgg16,
        tv_factory=tv_models.vgg16,
        tv_weights=tv_models.VGG16_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="vgg16_bn",
        eqx_factory=vgg16_bn,
        tv_factory=tv_models.vgg16_bn,
        tv_weights=tv_models.VGG16_BN_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="mobilenet_v2",
        eqx_factory=mobilenet_v2,
        tv_factory=tv_models.mobilenet_v2,
        tv_weights=tv_models.MobileNet_V2_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="mobilenet_v3_large",
        eqx_factory=mobilenet_v3_large,
        tv_factory=tv_models.mobilenet_v3_large,
        tv_weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="efficientnet_b0",
        eqx_factory=efficientnet_b0,
        tv_factory=tv_models.efficientnet_b0,
        tv_weights=tv_models.EfficientNet_B0_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="densenet121",
        eqx_factory=densenet121,
        tv_factory=tv_models.densenet121,
        tv_weights=tv_models.DenseNet121_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="googlenet",
        eqx_factory=googlenet,
        tv_factory=tv_models.googlenet,
        tv_weights=tv_models.GoogLeNet_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="shufflenetv2_x1.0",
        eqx_factory=shufflenet_v2_x1_0,
        tv_factory=tv_models.shufflenet_v2_x1_0,
        tv_weights=tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="regnet_y_400mf",
        eqx_factory=regnet_y_400mf,
        tv_factory=tv_models.regnet_y_400mf,
        tv_weights=tv_models.RegNet_Y_400MF_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="squeezenet1_0",
        eqx_factory=squeezenet1_0,
        tv_factory=tv_models.squeezenet1_0,
        tv_weights=tv_models.SqueezeNet1_0_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="convnext_tiny",
        eqx_factory=convnext_tiny,
        tv_factory=tv_models.convnext_tiny,
        tv_weights=tv_models.ConvNeXt_Tiny_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    ModelSpec(
        name="alexnet",
        eqx_factory=alexnet,
        tv_factory=tv_models.alexnet,
        tv_weights=tv_models.AlexNet_Weights.DEFAULT,
        xfail=_LEGACY_LOADER,
    ),
    # --- Segmentation models (legacy loader) ---
    ModelSpec(
        name="deeplabv3_resnet50",
        eqx_factory=deeplabv3,
        tv_factory=tv_models.segmentation.deeplabv3_resnet50,
        tv_weights=tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
        is_segmentation=True,
        xfail=_LEGACY_LOADER,
        eqx_kwargs={
            "backbone": resnet50(
                replace_stride_with_dilation=[False, True, True],
            ),
            "intermediate_layers": lambda x: [x.layer3, x.layer4],
            "aux_in_channels": 1024,
        },
    ),
    ModelSpec(
        name="fcn_resnet50",
        eqx_factory=fcn,
        tv_factory=tv_models.segmentation.fcn_resnet50,
        tv_weights=tv_models.segmentation.FCN_ResNet50_Weights.DEFAULT,
        is_segmentation=True,
        xfail=_LEGACY_LOADER,
        eqx_kwargs={
            "backbone": resnet50(
                replace_stride_with_dilation=[False, True, True],
            ),
            "intermediate_layers": lambda x: [x.layer3, x.layer4],
            "aux_in_channels": 1024,
        },
    ),
    ModelSpec(
        name="lraspp_mobilenetv3_large",
        eqx_factory=lraspp_mobilenet_v3_large,
        tv_factory=tv_models.segmentation.lraspp_mobilenet_v3_large,
        tv_weights=tv_models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        is_segmentation=True,
        xfail=_LEGACY_LOADER,
        eqx_kwargs={
            "backbone": mobilenet_v3_large(dilated=True),
            "intermediate_layers": lambda x: [4, 16],
        },
    ),
]


def _get_tv_weights_url(spec: ModelSpec) -> str:
    return spec.tv_weights.url


def _is_stateful(model) -> bool:
    return isinstance(model, eqx.nn.StatefulLayer) and model.is_stateful()


def _load_eqx_model(spec: ModelSpec):
    """Load eqxvision model using torchvision weights URL."""
    url = _get_tv_weights_url(spec)
    model = spec.eqx_factory(torch_weights=url, **spec.eqx_kwargs)
    model = eqx.nn.inference_mode(model)
    return model


def _load_tv_model(spec: ModelSpec):
    return spec.tv_factory(weights=spec.tv_weights, **spec.tv_kwargs).eval()


def _run_eqx_model(spec: ModelSpec, eqx_model, jax_input):
    """Run eqxvision model, returning output array."""
    if _is_stateful(eqx_model):
        state = eqx.nn.State(eqx_model)
        out, _state = eqx_model(jax_input, state, key=jr.PRNGKey(0))
    else:
        out = eqx_model(jax_input, key=jr.PRNGKey(0))
    if spec.is_segmentation:
        out = out[1]  # (aux, main) tuple -> main
    return out


def _run_tv_model(spec: ModelSpec, tv_model, torch_input):
    """Run torchvision model, returning output numpy array."""
    tv_out = tv_model(torch_input)
    if spec.is_segmentation:
        return tv_out["out"].squeeze(0).detach().numpy()
    return tv_out.squeeze(0).detach().numpy()


def _maybe_xfail(spec: ModelSpec):
    if spec.xfail is not None:
        pytest.xfail(spec.xfail)


@pytest.fixture(scope="session")
def random_input():
    np_img = np.random.RandomState(42).randn(3, 224, 224).astype(np.float32)
    jax_input = jnp.array(np_img)
    torch_input = torch.from_numpy(np_img.copy()).unsqueeze(0)
    return jax_input, torch_input


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: s.name)
def test_output_match(spec, random_input):
    _maybe_xfail(spec)
    jax_input, torch_input = random_input

    eqx_model = _load_eqx_model(spec)
    tv_model = _load_tv_model(spec)

    with torch.no_grad():
        tv_out = _run_tv_model(spec, tv_model, torch_input)

    eqx_out = _run_eqx_model(spec, eqx_model, jax_input)

    np.testing.assert_allclose(np.array(eqx_out), tv_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: s.name)
def test_gradient_match(spec, random_input):
    _maybe_xfail(spec)
    jax_input, torch_input_orig = random_input
    torch_input = torch_input_orig.clone().requires_grad_(True)

    eqx_model = _load_eqx_model(spec)
    tv_model = _load_tv_model(spec)

    # Torchvision gradient
    tv_out = tv_model(torch_input)
    if spec.is_segmentation:
        loss_tv = tv_out["out"].sum()
    else:
        loss_tv = tv_out.sum()
    loss_tv.backward()
    tv_grad = torch_input.grad.squeeze(0).numpy()

    # Eqxvision gradient
    def forward(x):
        return _run_eqx_model(spec, eqx_model, x).sum()

    eqx_grad = jax.grad(forward)(jax_input)

    np.testing.assert_allclose(np.array(eqx_grad), tv_grad, atol=1e-3, rtol=1e-3)
