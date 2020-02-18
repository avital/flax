# Polyak Averaging (HOWTO)

With Polyak Averaging, the training loop keeps track of additional parameters, 
which are the exponential moving average of the parameters over the course of 
training. Then, making predictions with these EMA parameters typically leads to 
better predictions.

([Link to diff view](https://github.com/marcvanzee/flax/compare/prerelease..howto-polyak-averaging?diff=split)
```diff
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 00a4017..8bce71f 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -12,7 +12,7 @@
 # See the License for the specific language governing permissions and
 # limitations under the License.
 
-
+# POLYAK DIFF
 
 
 """MNIST example.
diff --git a/markdown_generation/generate_markdown.py b/markdown_generation/generate_markdown.py
index 3493a3b..41cd760 100644
--- a/markdown_generation/generate_markdown.py
+++ b/markdown_generation/generate_markdown.py
@@ -44,7 +44,7 @@ flags.DEFINE_string('output_suffix', '.md',
                     'Markdown file.')
 
 flags.DEFINE_bool('exclude_tests', True, 'If true, do not show test files in '
-                   'diffs.')
+                  'diffs.')
 
 
 def wrap_in_code(lines, lang):
@@ -110,6 +110,8 @@ def generate_markdown_recursively():
     for filename in files:
       if not filename.endswith(FLAGS.input_suffix):
         continue
+      if FLAGS.exclude_tests and filename.endswith('_test.py'):
+        continue
       input_path = os.path.join(root, filename)
       output_path = prepare_output_path(root, filename)
       with open(output_path, 'w', encoding='utf8') as f:
```