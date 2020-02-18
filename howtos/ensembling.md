# Ensembling (HOWTO)

This howto changes the MNIST example from training a single CNNs to training an
ensemble of CNNs, such that each CNN is trained on its own device. Each CNN 
reports the accuracy and loss.

([Link to diff view](https://github.com/marcvanzee/flax/compare/prerelease..howto-ensembling?diff=split)
```diff
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index 00a4017..fe70dad 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -12,7 +12,7 @@
 # See the License for the specific language governing permissions and
 # limitations under the License.
 
-
+# ENSEMBLING DIFF
 
 
 """MNIST example.
```