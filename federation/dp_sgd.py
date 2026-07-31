"""
dp_sgd.py -- manual, Keras-3-native DP-SGD (NO tensorflow_privacy).

tensorflow_privacy's DPKeras* optimizers depend on keras.optimizers.legacy, which
Keras 3 removed, so they cannot run in the NGC container. This implements DP-SGD
directly via a custom keras.Model.train_step:

  1. per-example gradients  (num_microbatches = batch size => TRUE per-example)
  2. clip each example's gradient to global L2 norm C
  3. sum the clipped gradients, add Gaussian noise std = z*C
  4. divide by batch size and apply

Used ONLY for example-level (record-level) local DP. Non-DP and client-level
paths keep a plain keras.Model and are untouched.

Noise is reproducible and independent per client: a stateless RNG seeded by
(seed_base, step, var_index), with seed_base = f(SEED, client_id) set at build.

Two Keras-3 details this file must handle explicitly, because a custom train_step
opts out of the machinery compile() normally provides:

RANK ALIGNMENT. keras.losses.get("binary_crossentropy") returns the bare loss
FUNCTION, which requires rank(y_true) == rank(y_pred). compile(loss=...) instead
builds a Loss CLASS whose __call__ runs squeeze_or_expand_to_same_rank first, so
compile() reconciles a (N,) label vector against a (N,1) sigmoid output while the
raw function raises. The per-example loss here is resolved separately from
compile(), so it must reproduce that alignment itself -- otherwise every dataset
whose y is (N,) (body_signal_of_smoking) fails on the first batch, on BOTH the
vectorized and map_fn paths. Keras' LossFunctionWrapper is not public API, so
_align_ranks mirrors keras.src.losses.loss.squeeze_or_expand_to_same_rank.

METRIC STATE. Keras owns two stateful objects that its own train_step updates and
a custom one does not: `_loss_tracker` (the Mean behind the reported "loss") and
`_compile_metrics`. Leaving them untouched means, at epoch end,
_get_metrics_result_or_logs() -> get_metrics_result() either reports loss = 0.0
forever (model compiled without metrics) or raises "Cannot get result() since the
metric has not yet been built" (model compiled WITH metrics -- i.e. every model in
model.py). train_step therefore updates the tracker and routes metrics through
compute_metrics(), which costs one extra full-batch forward pass per step --
negligible against B per-example backward passes, and it keeps example-level fit
metrics identical in shape to every other mechanism's.

DESERIALIZATION. This is a Functional subclass, so a saved .keras is a functional
graph tagged DPSGDModel; on load Keras rebuilds the graph and calls
`DPSGDModel(inputs, outputs, name=...)` -- the functional path passes NO custom
__init__ args and ignores extra config keys (so a from_config override cannot
inject them). The DP hyperparameters therefore MUST be optional, or every export
fails to reload with "missing 3 required keyword-only arguments". They default to
an inert, no-op configuration (noise_multiplier=0.0, loss_fn=None): training
always constructs via make_dp_sgd_model(), which passes the real values, so the
defaults only ever apply at reload time -- where DP is irrelevant because the
attack pipeline does inference (predict/call) only and never runs train_step.
"""
import tensorflow as tf
from tensorflow import keras


def _align_ranks(y_true, y_pred):
    """Mirror of keras.src.losses.loss.squeeze_or_expand_to_same_rank.

    Applied before the per-example loss so DPSGDModel's gradient loss is the same
    quantity compile() computes. Only touches the last axis when it has size 1, so
    (1,) vs (1,10) -- sparse integer labels against class probabilities -- is left
    alone and still raises rather than being silently mangled.
    """
    t_rank = len(y_true.shape)
    p_rank = len(y_pred.shape)
    if t_rank == p_rank:
        return y_true, y_pred
    if t_rank == p_rank + 1 and y_true.shape[-1] == 1:
        if p_rank == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)
        else:
            y_true = tf.squeeze(y_true, axis=-1)
    elif p_rank == t_rank + 1 and y_pred.shape[-1] == 1:
        if t_rank == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        else:
            y_pred = tf.squeeze(y_pred, axis=-1)
    return y_true, y_pred


class DPSGDModel(keras.Model):
    # DP args are keyword-only AND optional. Optional is required for reload: the
    # functional deserialization path calls DPSGDModel(inputs, outputs, name=...)
    # with none of them (see module docstring). The inert defaults never affect
    # training -- make_dp_sgd_model() always passes explicit values -- they only
    # let an exported model reconstruct for inference in the attack pipeline.
    def __init__(self, *args,
                 l2_norm_clip: float = 1.0,
                 noise_multiplier: float = 0.0,
                 loss_fn=None,
                 seed_base: int = 0,
                 use_vectorized: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_norm_clip = float(l2_norm_clip)
        self.noise_multiplier = float(noise_multiplier)
        # a per-example loss callable; called on ONE example -> scalar.
        # A Loss instance is accepted too (it aligns ranks itself, so the
        # _align_ranks call in single() is then a no-op). None is accepted for
        # the reload path (inference only, train_step never runs).
        self._loss_fn = keras.losses.get(loss_fn) if isinstance(loss_fn, str) else loss_fn
        self._seed_base = int(seed_base)
        self._use_vectorized = bool(use_vectorized)
        self._dp_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="dp_step")

    # --- per-example gradients over the batch --------------------------------
    def _per_example_grads(self, x, y):
        variables = self.trainable_variables

        def single(inp):
            xi, yi = inp
            xi = tf.expand_dims(xi, 0)
            yi = tf.expand_dims(yi, 0)
            with tf.GradientTape() as tape:
                pred = self(xi, training=True)
                yt, yp = _align_ranks(yi, pred)               # match compile()
                loss = tf.reduce_mean(self._loss_fn(yt, yp))  # this example's loss
            g = tape.gradient(loss, variables)
            g = [tf.zeros_like(v) if gi is None else gi for gi, v in zip(g, variables)]
            return g, loss

        if self._use_vectorized:
            # fast path; if a layer breaks vectorization, set use_vectorized=False
            # to fall back to the sequential map_fn path. Recurrent stacks are the
            # known case: pfor cannot convert the GRU's TensorList (variant) ops and
            # fails with "`dtype` <dtype: 'variant'> is not compatible with 0 of
            # dtype int64", so the GRU datasets must run with use_vectorized=False.
            grads, losses = tf.vectorized_map(single, (x, y))
        else:
            sig = ([tf.TensorSpec(v.shape, v.dtype) for v in variables],
                   tf.TensorSpec([], tf.float32))
            grads, losses = tf.map_fn(single, (x, y), fn_output_signature=sig)
        return grads, losses

    def train_step(self, data):
        x, y = data[0], data[1]
        variables = self.trainable_variables
        B = tf.shape(x)[0]
        batch_f = tf.cast(B, tf.float32)

        per_ex_grads, per_ex_losses = self._per_example_grads(x, y)
        # per_ex_grads: list over variables, each shaped [B, *var.shape]

        # per-example global gradient norm (across ALL variables)
        sq = tf.zeros([B], dtype=tf.float32)
        for g in per_ex_grads:
            sq += tf.reduce_sum(tf.square(tf.reshape(g, [B, -1])), axis=1)
        norms = tf.sqrt(sq) + 1e-12
        scale = tf.minimum(1.0, self.l2_norm_clip / norms)          # [B]

        step = tf.identity(self._dp_step)
        self._dp_step.assign_add(1)
        stddev = self.noise_multiplier * self.l2_norm_clip

        new_grads = []
        for i, g in enumerate(per_ex_grads):
            rank = len(g.shape)
            scale_b = tf.reshape(scale, tf.concat([[B], tf.ones([rank - 1], tf.int32)], 0))
            summed = tf.reduce_sum(g * scale_b, axis=0)             # sum of clipped grads
            seed = tf.stack([tf.cast(self._seed_base + i, tf.int64), step])
            noise = tf.random.stateless_normal(
                tf.shape(summed), seed=seed, stddev=stddev, dtype=summed.dtype)
            new_grads.append((summed + noise) / batch_f)           # average

        self.optimizer.apply_gradients(zip(new_grads, variables))

        # --- keep Keras' own metric state in sync (see module docstring) ------
        loss_mean = tf.reduce_mean(per_ex_losses)
        tracker = getattr(self, "_loss_tracker", None)
        if tracker is not None:
            tracker.update_state(loss_mean)
        if getattr(self, "_compile_metrics", None) is not None:
            # compute_metrics() builds + updates the compiled metrics and returns
            # every metric result, including the tracker's "loss" updated above.
            y_pred = self(x, training=True)
            return self.compute_metrics(x, y, y_pred)
        return {"loss": loss_mean if tracker is None else tracker.result()}


def make_dp_sgd_model(inputs, outputs, name, *,
                      l2_norm_clip, noise_multiplier, loss_fn,
                      seed_base=0, use_vectorized=True):
    """Functional-API construction of a DP-SGD model (same graph, DP train_step)."""
    return DPSGDModel(
        inputs=inputs, outputs=outputs, name=name,
        l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier,
        loss_fn=loss_fn, seed_base=seed_base, use_vectorized=use_vectorized,
    )
