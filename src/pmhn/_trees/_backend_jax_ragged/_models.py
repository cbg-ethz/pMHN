from __future__ import annotations

import jax
import jax.numpy as jnp

try:
    import equinox as eqx
except ModuleNotFoundError:  # pragma: no cover - exercised only when missing dep
    eqx = None


if eqx is not None:

    class BilinearLogRateModel(eqx.Module):
        """Bilinear model in the same functional form as the original backend."""

        theta: jax.Array

        def __call__(
            self, lineage_multihot: jax.Array, new_mut: jax.Array
        ) -> jax.Array:
            # new_mut is 1-indexed.
            return jnp.sum(self.theta[new_mut - 1] * lineage_multihot)

    class MLPLogRateModel(eqx.Module):
        """Simple Equinox MLP model for generic log-rates."""

        mlp: eqx.nn.MLP
        n_genes: int = eqx.field(static=True)

        def __init__(
            self,
            n_genes: int,
            key: jax.Array,
            width_size: int = 64,
            depth: int = 2,
        ) -> None:
            self.n_genes = n_genes
            self.mlp = eqx.nn.MLP(
                in_size=2 * n_genes,
                out_size=1,
                width_size=width_size,
                depth=depth,
                activation=jax.nn.tanh,
                final_activation=lambda x: x,
                key=key,
            )

        def __call__(
            self, lineage_multihot: jax.Array, new_mut: jax.Array
        ) -> jax.Array:
            # new_mut is 1-indexed.
            new_mut_onehot = jax.nn.one_hot(
                new_mut - 1, self.n_genes, dtype=jnp.float32
            )
            features = jnp.concatenate(
                [lineage_multihot.astype(jnp.float32), new_mut_onehot],
                axis=-1,
            )
            return jnp.squeeze(self.mlp(features), axis=-1)

else:

    class BilinearLogRateModel:  # pragma: no cover - exercised only when missing dep
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "equinox is required to use BilinearLogRateModel. "
                "Install equinox or use the bilinear theta backend."
            )

    class MLPLogRateModel:  # pragma: no cover - exercised only when missing dep
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "equinox is required to use MLPLogRateModel. "
                "Install equinox and retry."
            )
