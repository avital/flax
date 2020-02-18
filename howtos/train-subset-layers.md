# Train only a Few Layers (HOWTO)

Goal: Create a model from which only the last (few) layers are trained.

Implementation: This uses a base model (CNN), which outputs both the logits and
a representation_layer. This is used by a refinement model (RefineCNN). Only the
latter is trained (ie used in the optimizer). The following lines are
(hopefully) illustrative:
```
  base_logits, base_representation = base_model(batch['image'])
  logits = refine_model(base_representation)
  ...
  optimizer = create_optimizer(refine_model, … )
```

([Link to diff view](https://github.com/marcvanzee/flax/compare/prerelease..howto-train-subset-layers?diff=split)
```diff
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 020838b..7e89463 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -65,22 +65,37 @@ class CNN(nn.Module):
 
   def apply(self, x):
     x = nn.Conv(x, features=32, kernel_size=(3, 3))
-    x = nn.relu(x)
+    x = jax.nn.relu(x)
     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
     x = nn.Conv(x, features=64, kernel_size=(3, 3))
-    x = nn.relu(x)
+    x = jax.nn.relu(x)
     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
     x = x.reshape((x.shape[0], -1))  # flatten
-    x = nn.Dense(x, features=256)
-    x = nn.relu(x)
+    x = nn.Dense(x, features=256, name='representation')
+    x = jax.nn.relu(x)
+    penultimate_layer = x
     x = nn.Dense(x, features=10)
-    x = nn.log_softmax(x)
+    x = jax.nn.log_softmax(x)
+    return x, penultimate_layer
+
+
+class RefineCNN(nn.Module):
+  """Refine CNN Model."""
+
+  def apply(self, x):
+    x = nn.Dense(x, features=64)
+    x = jax.nn.relu(x)
+    x = nn.Dense(x, features=10, name='final')
+    x = jax.nn.log_softmax(x)
     return x
 
 
 def create_model(key):
-  _, model = CNN.create_by_shape(key, [((1, 28, 28, 1), jnp.float32)])
-  return model
+  model_def = CNN()
+  _, model = model_def.create_by_shape(key, [((1, 28, 28, 1), jnp.float32)])
+  refine_def = RefineCNN()
+  _, refine_model = refine_def.create_by_shape(key, [((1, 256), jnp.float32)])
+  return model, refine_model
 
 
 def create_optimizer(model, learning_rate, beta):
@@ -109,10 +124,11 @@ def compute_metrics(logits, labels):
 
 
 @jax.jit
-def train_step(optimizer, batch):
+def train_step(optimizer, base_model, batch):
   """Train for a single step."""
   def loss_fn(model):
-    logits = model(batch['image'])
+    _, base_representation = base_model(batch['image'])
+    logits = model(base_representation)
     loss = cross_entropy_loss(logits, batch['label'])
     return loss, logits
   optimizer, _, logits = optimizer.optimize(loss_fn)
@@ -121,23 +137,25 @@ def train_step(optimizer, batch):
 
 
 @jax.jit
-def eval_step(model, batch):
-  logits = model(batch['image'])
+def eval_step(model, base_model, batch):
+  _, base_representation = base_model(batch['image'])
+  logits = model(base_representation)
   return compute_metrics(logits, batch['label'])
 
 
-def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
+
+def train_epoch(optimizer, base_model, train_ds, batch_size, epoch):
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
+    optimizer, metrics = train_step(optimizer, base_model, batch)
     batch_metrics.append(metrics)
 
   # compute mean of metrics across each batch in epoch.
@@ -151,9 +169,8 @@ def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
 
   return optimizer, epoch_metrics_np
 
-
-def eval_model(model, test_ds):
-  metrics = eval_step(model, test_ds)
+def eval_model(model, base_model, test_ds):
+  metrics = eval_step(model, base_model, test_ds)
   metrics = jax.device_get(metrics)
   summary = jax.tree_map(lambda x: x.item(), metrics)
   return summary['loss'], summary['accuracy']
@@ -173,18 +190,27 @@ def train(train_ds, test_ds):
   batch_size = FLAGS.batch_size
   num_epochs = FLAGS.num_epochs
 
-  model = create_model(rng)
-  optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)
+  base_model, refine_model = create_model(rng)
+  optimizer = create_optimizer(refine_model,
+                               FLAGS.learning_rate, FLAGS.momentum)
 
-  input_rng = onp.random.RandomState(0)
+  print('Base model weights:')
+  print(base_model.params['representation']['kernel'][0:5, 0:5])
+  print('Refine model weights:')
+  print(refine_model.params['final']['kernel'][0:5, 0:5])
 
   for epoch in range(1, num_epochs + 1):
-    optimizer, _ = train_epoch(
-        optimizer, train_ds, batch_size, epoch, input_rng)
-    loss, accuracy = eval_model(optimizer.target, test_ds)
+    optimizer = train_epoch(optimizer, base_model, train_ds, batch_size, epoch)
+    loss, accuracy = eval_model(optimizer.target, base_model, test_ds)
     logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                  epoch, loss, accuracy * 100)
-  return optimizer
+
+  refine_model = optimizer.target
+  print('Base model weights should be the same:')
+  print(base_model.params['representation']['kernel'][0:5, 0:5])
+  print('Refine model weights should be changed:')
+  print(refine_model.params['final']['kernel'][0:5, 0:5])
+  print('-----')
 
 
 def main(_):
```