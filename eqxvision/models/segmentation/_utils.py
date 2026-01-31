from typing import Optional

import equinox as eqx
import jax
import jax.image as jim
import jax.random as jr
from equinox._custom_types import sentinel
from jaxtyping import Array


class _SimpleSegmentationModel(eqx.nn.StatefulLayer):
    backbone: eqx.Module
    classifier: eqx.Module
    aux_classifier: eqx.Module

    def __init__(
        self,
        backbone: "eqx.Module",
        classifier: "eqx.Module",
        aux_classifier: Optional["eqx.Module"] = None,
    ) -> None:
        """

        **Arguments:**

        - `backbone`: the network used to compute the features for the model
            The backbone returns `embedding_features(Ignored)`, `[output features of intermediate layers]`.
        - `classifier`: module that takes last of the intermediate outputs from the
            backbone and returns a dense prediction
        - `aux_classifier`: If used, an auxiliary classifier similar to `classifier` for the auxiliary layer
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def __call__(
        self, x: Array, state: eqx.nn.State = sentinel, *, key: "jax.random.PRNGKey"
    ):
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels
        - `state`: An `eqx.nn.State` object for batch norm running statistics
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`

        **Returns:**
        A tuple of ((aux_output, output), state).
        """
        keys = jr.split(key, 3)
        if state is not sentinel:
            (_, xs), state = self.backbone(x, state=state, key=keys[0])
            x_clf, state = self.classifier(xs[-1], state=state, key=keys[1])
        else:
            (_, xs) = self.backbone(x, key=keys[0])
            x_clf = self.classifier(xs[-1], key=keys[1])
        target_shape = (x_clf.shape[0], x.shape[-2], x.shape[-1])
        x_clf = jim.resize(x_clf, shape=target_shape, method="bilinear")

        if self.aux_classifier is not None:
            if state is not sentinel:
                x_aux, state = self.aux_classifier(xs[0], state=state, key=keys[2])
            else:
                x_aux = self.aux_classifier(xs[0], key=keys[2])
            target_shape = (x_aux.shape[0], x.shape[-2], x.shape[-1])
            x_aux = jim.resize(x_aux, shape=target_shape, method="bilinear")
            if state is not sentinel:
                return (x_aux, x_clf), state
            return (x_aux, x_clf)

        if state is not sentinel:
            return (None, x_clf), state
        return (None, x_clf)
