diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 020838b..deeb610 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -17,6 +17,8 @@ This script trains a simple Convolutional Neural Net on the MNIST dataset.
 The data is loaded using tensorflow_datasets.
 """
 
+import functools
+
 from absl import app
 from absl import flags
 from absl import logging
@@ -83,9 +85,11 @@ def create_model(key):
   return model
 
 
-def create_optimizer(model, learning_rate, beta):
-  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
-  optimizer = optimizer_def.create(model)
+@jax.pmap
+def create_optimizers(rng):
+  optimizer_def = optim.Momentum(
+      learning_rate=FLAGS.learning_rate, beta=FLAGS.momentum)
+  optimizer = optimizer_def.create(create_model(rng))
   return optimizer
 
 
@@ -108,7 +112,7 @@ def compute_metrics(logits, labels):
   return metrics
 
 
-@jax.jit
+@functools.partial(jax.pmap)
 def train_step(optimizer, batch):
   """Train for a single step."""
   def loss_fn(model):
@@ -120,13 +124,17 @@ def train_step(optimizer, batch):
   return optimizer, metrics
 
 
-@jax.jit
+@jax.pmap
 def eval_step(model, batch):
   logits = model(batch['image'])
   return compute_metrics(logits, batch['label'])
 
 
-def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
+def replicate(tree_obj, num_replicas):
+  return jax.tree_map(lambda x: onp.array([x] * num_replicas), tree_obj)
+
+
+def train_epoch(optimizers, train_ds, batch_size, epoch, rng, num_models):
   """Train for a single epoch."""
   train_ds_size = len(train_ds['image'])
   steps_per_epoch = train_ds_size // batch_size
@@ -137,25 +145,28 @@ def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
   batch_metrics = []
   for perm in perms:
     batch = {k: v[perm] for k, v in train_ds.items()}
-    optimizer, metrics = train_step(optimizer, batch)
+    batch = replicate(batch, num_models)
+    optimizers, metrics = train_step(optimizers, batch)
     batch_metrics.append(metrics)
 
   # compute mean of metrics across each batch in epoch.
   batch_metrics_np = jax.device_get(batch_metrics)
+  batch_metrics_np = jax.tree_multimap(lambda *xs: onp.array(xs),
+                                       *batch_metrics_np)
   epoch_metrics_np = {
-      k: onp.mean([metrics[k] for metrics in batch_metrics_np])
-      for k in batch_metrics_np[0]}
+      k: onp.mean(batch_metrics_np[k], axis=0) for k in batch_metrics_np
+  }
 
   logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)
 
-  return optimizer, epoch_metrics_np
+  return optimizers, epoch_metrics_np
 
 
-def eval_model(model, test_ds):
-  metrics = eval_step(model, test_ds)
+def eval_model(models, test_ds):
+  metrics = eval_step(models, test_ds)
   metrics = jax.device_get(metrics)
-  summary = jax.tree_map(lambda x: x.item(), metrics)
+  summary = metrics
   return summary['loss'], summary['accuracy']
 
 
@@ -172,19 +183,20 @@ def train(train_ds, test_ds):
 
   batch_size = FLAGS.batch_size
   num_epochs = FLAGS.num_epochs
+  num_models = jax.device_count()
 
-  model = create_model(rng)
-  optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)
+  optimizers = create_optimizers(random.split(rng, num_models))
 
   input_rng = onp.random.RandomState(0)
+  test_ds = replicate(test_ds, num_models)
 
   for epoch in range(1, num_epochs + 1):
-    optimizer, _ = train_epoch(
-        optimizer, train_ds, batch_size, epoch, input_rng)
-    loss, accuracy = eval_model(optimizer.target, test_ds)
-    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
-                 epoch, loss, accuracy * 100)
-  return optimizer
+    optimizers, _ = train_epoch(optimizers, train_ds, batch_size, epoch,
+                                input_rng, num_models)
+    loss, accuracy = eval_model(optimizers.target, test_ds)
+    logging.info('eval epoch: %d, loss: %s, accuracy: %s', epoch, loss,
+                 accuracy * 100)
+  return optimizers
 
 
 def main(_):
diff --git a/examples/mnist/train_test.py b/examples/mnist/train_test.py
index 5b1dbf0..76ef579 100644
--- a/examples/mnist/train_test.py
+++ b/examples/mnist/train_test.py
@@ -29,13 +29,12 @@ class TrainTest(absltest.TestCase):
   def test_train_one_epoch(self):
     train_ds, test_ds = train.get_datasets()
     input_rng = onp.random.RandomState(0)
-    model = train.create_model(random.PRNGKey(0))
-    optimizer = train.create_optimizer(model, 0.1, 0.9)
-    optimizer, train_metrics = train.train_epoch(optimizer, train_ds, 128, 0,
-                                                 input_rng)
+    optimizers = train.create_optimizers(random.split(random.PRNGKey(0), 1))
+    optimizers, train_metrics = train.train_epoch(optimizers, train_ds, 128, 0,
+                                                  input_rng, 1)
     self.assertLessEqual(train_metrics['loss'], 0.27)
     self.assertGreaterEqual(train_metrics['accuracy'], 0.92)
-    loss, accuracy = train.eval_model(optimizer.target, test_ds)
+    loss, accuracy = train.eval_model(optimizers.target, test_ds, 1)
     self.assertLessEqual(loss, 0.06)
     self.assertGreaterEqual(accuracy, 0.98)
 
