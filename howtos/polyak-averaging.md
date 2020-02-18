# Polyak Averaging (HOWTO)

With Polyak Averaging, the training loop keeps track of additional parameters, 
which are the exponential moving average of the parameters over the course of 
training. Then, making predictions with these EMA parameters typically leads to 
better predictions.

([Full diff view](https://github.com/marcvanzee/flax/compare/prerelease..howto-polyak-averaging?diff=split))
```diff
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 020838b..46eaab2 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -109,15 +109,19 @@ def compute_metrics(logits, labels):
 
 
 @jax.jit
-def train_step(optimizer, batch):
+def train_step(optimizer, params_ema, batch):
   """Train for a single step."""
   def loss_fn(model):
     logits = model(batch['image'])
     loss = cross_entropy_loss(logits, batch['label'])
     return loss, logits
   optimizer, _, logits = optimizer.optimize(loss_fn)
+  params_ema = jax.tree_multimap(
+      lambda p_ema, p: p_ema * 0.99 + p * 0.01,
+      params_ema, optimizer.target.params)
+
   metrics = compute_metrics(logits, batch['label'])
-  return optimizer, metrics
+  return optimizer, params_ema, metrics
 
 
 @jax.jit
@@ -126,18 +130,18 @@ def eval_step(model, batch):
   return compute_metrics(logits, batch['label'])
 
 
-def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
+def train_epoch(optimizer, params_ema, train_ds, batch_size, epoch):
   """Train for a single epoch."""
   train_ds_size = len(train_ds['image'])
   steps_per_epoch = train_ds_size // batch_size
 
-  perms = rng.permutation(len(train_ds['image']))
+  perms = onp.random.permutation(len(train_ds['image']))
   perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
   perms = perms.reshape((steps_per_epoch, batch_size))
   batch_metrics = []
   for perm in perms:
     batch = {k: v[perm] for k, v in train_ds.items()}
-    optimizer, metrics = train_step(optimizer, batch)
+    optimizer, params_ema, metrics = train_step(optimizer, params_ema, batch)
     batch_metrics.append(metrics)
 
   # compute mean of metrics across each batch in epoch.
@@ -175,15 +179,22 @@ def train(train_ds, test_ds):
 
   model = create_model(rng)
   optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)
+  params_ema = model.params
 
   input_rng = onp.random.RandomState(0)
 
   for epoch in range(1, num_epochs + 1):
     optimizer, _ = train_epoch(
-        optimizer, train_ds, batch_size, epoch, input_rng)
+        optimizer, params_ema, train_ds, batch_size, epoch, input_rng)
     loss, accuracy = eval_model(optimizer.target, test_ds)
     logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                  epoch, loss, accuracy * 100)
+
+    model_ema = optimizer.target.replace(params=params_ema)
+    polyak_loss, polyak_accuracy = eval_model(model_ema, test_ds)
+    logging.info('polyak eval epoch: %d, loss: %.4f, accuracy: %.2f',
+                 epoch, polyak_loss, polyak_accuracy * 100)
+
   return optimizer
 
 
```