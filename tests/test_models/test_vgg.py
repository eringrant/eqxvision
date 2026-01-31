import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


model_list = [
    ("vgg11", models.vgg11),
    ("vgg11_bn", models.vgg11_bn),
]


class TestVGG:
    @pytest.mark.parametrize("model_func", model_list)
    def test_pretrained(self, getkey, model_func, demo_image, net_preds):
        img = demo_image(224)
        keys = jax.random.split(getkey(), 1)

        @eqx.filter_jit
        def forward(net, state, imgs, keys):
            def fn(x, state, key):
                return net(x, state, key=key)

            outputs, state = jax.vmap(
                fn, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
            )(imgs, state, keys)
            return outputs

        model = model_func[1](torch_weights=CLASSIFICATION_URLS[model_func[0]])
        model = eqx.tree_inference(model, True)
        state = eqx.nn.State(model)
        pt_outputs = net_preds[model_func[0]]
        eqx_outputs = forward(model, state, img, keys)

        assert jnp.isclose(eqx_outputs, pt_outputs, atol=1e-4).all()
