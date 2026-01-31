import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import pytest

import eqxvision.models as models


# Models that use BatchNorm and need state threading
stateful_clf_list = [
    models.densenet121,
    models.efficientnet_b0,
    models.efficientnet_v2_s,
    models.googlenet,
    models.mobilenet_v2,
    models.mobilenet_v3_small,
    models.regnet_x_400mf,
    models.resnet18,
    models.shufflenet_v2_x0_5,
    models.vgg11_bn,
]

# Models that do NOT use BatchNorm (no state needed)
stateless_clf_list = [
    models.alexnet,
    models.convnext_tiny,
    models.squeezenet1_0,
    models.swin_t,
    models.swin_v2_t,
    models.vgg11,
    models.vit_tiny,
]


class TestGrads:
    @pytest.mark.parametrize("model_func", stateful_clf_list)
    def test_classification_stateful(self, model_func, getkey):
        @eqx.filter_value_and_grad
        def compute_loss(model, state, x, y):
            keys = jrandom.split(getkey(), x.shape[0])

            def fn(x, state, key):
                return model(x, state, key=key)

            output, state = jax.vmap(
                fn, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
            )(x, state, keys)
            if isinstance(output, tuple):
                output = output[0]
            one_hot_actual = jax.nn.one_hot(y, num_classes=3)
            return optax.softmax_cross_entropy(output, one_hot_actual).mean()

        @eqx.filter_jit
        def make_step(model, state, x, y, optimizer, opt_state):
            loss, grads = compute_loss(model, state, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        num_classes = 3
        net, state = eqx.nn.make_with_state(model_func)(num_classes=num_classes)
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

        random_image = jax.random.uniform(
            key=jax.random.PRNGKey(0), shape=(1, 3, 224, 224)
        )
        loss, net, _ = make_step(
            net, state, random_image, jnp.asarray([1]), optimizer, opt_state
        )

        assert not jnp.isnan(loss).any()

    @pytest.mark.parametrize("model_func", stateless_clf_list)
    def test_classification_stateless(self, model_func, getkey):
        @eqx.filter_value_and_grad
        def compute_loss(model, x, y):
            keys = jrandom.split(getkey(), x.shape[0])
            output = jax.vmap(model, axis_name="batch")(x, key=keys)
            one_hot_actual = jax.nn.one_hot(y, num_classes=3)
            return optax.softmax_cross_entropy(output, one_hot_actual).mean()

        @eqx.filter_jit
        def make_step(model, x, y, optimizer, opt_state):
            loss, grads = compute_loss(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        num_classes = 3
        net = model_func(num_classes=num_classes)
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

        if model_func == models.swin_v2_t:
            random_image = jax.random.uniform(
                key=jax.random.PRNGKey(0), shape=(1, 3, 256, 256)
            )
        else:
            random_image = jax.random.uniform(
                key=jax.random.PRNGKey(0), shape=(1, 3, 224, 224)
            )
        loss, net, _ = make_step(
            net, random_image, jnp.asarray([1]), optimizer, opt_state
        )

        assert not jnp.isnan(loss).any()
