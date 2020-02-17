# Polyak Averaging (HOWTO)

With Polyak Averaging, the training loop keeps track of additional parameters, 
which are the exponential moving average of the parameters over the course of 
training. Then, making predictions with these EMA parameters typically leads to 
better predictions.

```diff
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 52347aa..0e7624f 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -12,12 +12,11 @@
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
 
 from absl import app
@@ -112,15 +111,19 @@ def compute_metrics(logits, labels):
 
 
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
@@ -129,7 +132,7 @@ def eval_step(model, batch):
   return compute_metrics(logits, batch['label'])
 
 
-def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
+def train_epoch(optimizer, params_ema, train_ds, batch_size, epoch, rng):
   """Train for a single epoch."""
   train_ds_size = len(train_ds['image'])
   steps_per_epoch = train_ds_size // batch_size
@@ -140,7 +143,7 @@ def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
   batch_metrics = []
   for perm in perms:
     batch = {k: v[perm] for k, v in train_ds.items()}
-    optimizer, metrics = train_step(optimizer, batch)
+    optimizer, params_ema, metrics = train_step(optimizer, params_ema, batch)
     batch_metrics.append(metrics)
 
   # compute mean of metrics across each batch in epoch.
@@ -178,15 +181,21 @@ def train(train_ds, test_ds):
 
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
   return optimizer
 
 
@@ -196,4 +205,4 @@ def main(_):
 
 
 if __name__ == '__main__':
-  app.run(main)
\ No newline at end of file
+  app.run(main)
diff --git a/generate_templates.py b/generate_templates.py
index 439d9de..c7b53a8 100644
--- a/generate_templates.py
+++ b/generate_templates.py
@@ -8,38 +8,26 @@ FLAGS = flags.FLAGS
 
 flags.DEFINE_string('repo_path', '.', 'Relative path to the Github repository.')
 
-flags.DEFINE_bool('strip_copyright_from_code', True, 'If true, removes copyright header from code.')
-
 # TODO(marcvanzee): Currently this only works for README.md. Think of a more 
 # general solution.
 
-def wrap_in_code(lines, lang):
-  return [f'```{lang}'] + lines + ['```']
-
+def wrap_in_code(text, type='py'):
+  return [f'```{type}'] + text.split('\n') + ['```']
 
 def insert_branch_diff_fn(target_branch, source_branch='prerelease'):
   git = Repo(FLAGS.repo_path).git
   diff_text = git.diff(source_branch, target_branch, '--', ':(exclude)*.md')
-  return wrap_in_code(diff_text.split('\n'), 'diff')
+  return wrap_in_code(diff_text, 'diff')
 
 
 def insert_py_code_fn(file_path, lang='py'):
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
+  lines = open(file_path, encoding='utf8').readlines()
+  return wrap_in_code(lines)
 
 
 operations = {
   '@insert_branch_diff': insert_branch_diff_fn,
-  '@insert_code': insert_py_code_fn
+  '@insert_py_code': insert_py_code_fn
 }
 
 def generate_template(input_path, output_path):
@@ -54,6 +42,7 @@ def generate_template(input_path, output_path):
   with open(output_path, 'w', encoding='utf8') as output_f:
     output_f.write('\n'.join(output_lines))
 
+  
 
 def main(argv):
   if len(argv) > 1:
```