# Ensembling (HOWTO)

This howto changes the MNIST example from training a single CNNs to training an
ensemble of CNNs, such that each CNN is trained on its own device. Each CNN 
reports the accuracy and loss.

```diff
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 52347aa..7ac4fe2 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -12,13 +12,13 @@
 # See the License for the specific language governing permissions and
 # limitations under the License.
 
-
-
-
 """MNIST example.
+
 This script trains a simple Convolutional Neural Net on the MNIST dataset.
 The data is loaded using tensorflow_datasets.
+
 """
+import functools
 
 from absl import app
 from absl import flags
@@ -86,9 +86,11 @@ def create_model(key):
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
 
 
@@ -111,7 +113,7 @@ def compute_metrics(logits, labels):
   return metrics
 
 
-@jax.jit
+@functools.partial(jax.pmap)
 def train_step(optimizer, batch):
   """Train for a single step."""
   def loss_fn(model):
@@ -123,13 +125,17 @@ def train_step(optimizer, batch):
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
@@ -140,25 +146,27 @@ def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
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
-
-  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
+      k: onp.mean(batch_metrics_np[k], axis=0) for k in batch_metrics_np
+  }
+  logging.info('train epoch: %d, loss: %s, accuracy: %s', epoch,
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
 
 
@@ -175,19 +183,20 @@ def train(train_ds, test_ds):
 
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
@@ -196,4 +205,4 @@ def main(_):
 
 
 if __name__ == '__main__':
-  app.run(main)
\ No newline at end of file
+  app.run(main)
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
 
diff --git a/generate_templates.py b/generate_templates.py
deleted file mode 100644
index 439d9de..0000000
--- a/generate_templates.py
+++ /dev/null
@@ -1,66 +0,0 @@
-"""Test."""
-from absl import app
-from absl import flags
-
-from git import Repo
-
-FLAGS = flags.FLAGS
-
-flags.DEFINE_string('repo_path', '.', 'Relative path to the Github repository.')
-
-flags.DEFINE_bool('strip_copyright_from_code', True, 'If true, removes copyright header from code.')
-
-# TODO(marcvanzee): Currently this only works for README.md. Think of a more 
-# general solution.
-
-def wrap_in_code(lines, lang):
-  return [f'```{lang}'] + lines + ['```']
-
-
-def insert_branch_diff_fn(target_branch, source_branch='prerelease'):
-  git = Repo(FLAGS.repo_path).git
-  diff_text = git.diff(source_branch, target_branch, '--', ':(exclude)*.md')
-  return wrap_in_code(diff_text.split('\n'), 'diff')
-
-
-def insert_py_code_fn(file_path, lang='py'):
-  lines = open(file_path, encoding='utf8').read().splitlines()
-  if FLAGS.strip_copyright_from_code:
-    while len(lines) and lines[0].startswith('#'):
-      lines = lines[1:]
-  # Strips empty lines from start and aned.
-  while len(lines) and not lines[0].strip():
-    lines = lines[1:]
-  while len(lines) and not lines[-1].strip():
-    lines = lines[:-1]
-  
-  return wrap_in_code(lines, lang=lang)
-
-
-operations = {
-  '@insert_branch_diff': insert_branch_diff_fn,
-  '@insert_code': insert_py_code_fn
-}
-
-def generate_template(input_path, output_path):
-  output_lines = []
-  for line in open(input_path, encoding='utf8').read().splitlines():
-    words = line.split()
-    if len(words) < 2 or words[0] not in operations:
-      output_lines.append(line)
-      continue
-    output_lines += operations[words[0]](*words[1:])
-
-  with open(output_path, 'w', encoding='utf8') as output_f:
-    output_f.write('\n'.join(output_lines))
-
-
-def main(argv):
-  if len(argv) > 1:
-    raise app.UsageError('Too many command-line arguments.')
-  
-  generate_template('.README.template.md', 'README.md')
-
-
-if __name__ == '__main__':
-  app.run(main)
```