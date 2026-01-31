import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestRegNet:
    def test_pretrained(self, getkey, demo_image, net_preds):
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

        model = models.regnet_x_400mf(
            torch_weights=CLASSIFICATION_URLS["regnet_x_400mf"]
        )
        model = eqx.tree_inference(model, True)
        state = eqx.nn.State(model)
        eqx_outputs = forward(model, state, img, keys)
        pt_outputs = net_preds["regnet_x_400mf"]

        assert jnp.isclose(eqx_outputs, pt_outputs, atol=1e-4).all()
