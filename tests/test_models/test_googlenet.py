import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestGoogLeNet:
    answer = (1, 1000)

    def test_output_shape(self, demo_image, getkey):
        img = demo_image(224)

        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = models.googlenet(num_classes=1000, aux_logits=False)
        output = forward(model, img, getkey())
        assert output.shape == self.answer

    def test_pretrained(self, getkey, demo_image, net_preds):
        img = demo_image(224)

        @eqx.filter_jit
        def forward(net, state, x, key):
            keys = jax.random.split(key, x.shape[0])

            def fn(x_, state_, key_):
                return net(x_, state_, key=key_)

            ans, state = jax.vmap(
                fn, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
            )(x, state, keys)
            return ans

        model = models.googlenet(torch_weights=CLASSIFICATION_URLS["googlenet"])
        model = eqx.tree_inference(model, True)
        state = eqx.nn.State(model)
        eqx_outputs = forward(model, state, img, getkey())
        pt_outputs = net_preds["googlenet"]

        assert jnp.isclose(eqx_outputs, pt_outputs, atol=1e-4).all()
