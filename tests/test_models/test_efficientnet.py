import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


model_list = [
    ("efficientnet_b0", models.efficientnet_b0),
    ("efficientnet_v2_s", models.efficientnet_v2_s),
]


class TestEfficientNet:
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
        eqx_outputs = forward(model, state, img, keys)
        pt_outputs = net_preds[model_func[0]]

        assert jnp.argmax(eqx_outputs) == jnp.argmax(pt_outputs)
