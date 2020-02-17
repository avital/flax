# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for flax.examples.mnist.train."""

from absl.testing import absltest

from flax.examples.mnist import train
from jax import random

import numpy as onp


class TrainTest(absltest.TestCase):

  def test_train_one_epoch(self):
    train_ds, test_ds = train.get_datasets()
    input_rng = onp.random.RandomState(0)
    optimizers = train.create_optimizers(random.split(random.PRNGKey(0), 1))
    optimizers, train_metrics = train.train_epoch(optimizers, train_ds, 128, 0,
                                                  input_rng, 1)
    self.assertLessEqual(train_metrics['loss'], 0.27)
    self.assertGreaterEqual(train_metrics['accuracy'], 0.92)
    loss, accuracy = train.eval_model(optimizers.target, test_ds, 1)
    self.assertLessEqual(loss, 0.06)
    self.assertGreaterEqual(accuracy, 0.98)


if __name__ == '__main__':
  absltest.main()
