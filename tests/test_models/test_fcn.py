import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from eqxvision.models import fcn, resnet50
from eqxvision.utils import SEGMENTATION_URLS


@eqx.filter_jit
def forward(model, state, x, key):
    def fn(x_, state_, key_):
        return model(x_, state_, key=key_)

    (aux, clf), state = jax.vmap(
        fn, axis_name="batch", in_axes=(0, None, 0), out_axes=((0, 0), None)
    )(x, state, key)
    return aux, clf


def test_fcn(demo_image, net_preds):
    img = demo_image(224)
    net = fcn(
        backbone=resnet50(replace_stride_with_dilation=[False, True, True]),
        intermediate_layers=lambda x: [x.layer3, x.layer4],
        aux_in_channels=1024,
        torch_weights=SEGMENTATION_URLS["fcn_resnet50"],
    )
    net = eqx.tree_inference(net, True)
    state = eqx.nn.State(net)
    aux, out = forward(net, state, img, key=jr.split(jr.PRNGKey(0), 1))

    pt_outputs = net_preds["fcn_resnet50"]
    assert jnp.isclose(out, pt_outputs, atol=1e-4).all()
