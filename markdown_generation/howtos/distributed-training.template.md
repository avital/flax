# Distributed Training (HOWTO)

This howto changes the MNIST example from single-device training to multi-device
training on a single host. The evaluation is still performed on a single device. 

Note the code does not support multi-host distributed training.

@insert_branch_diff howto-distributed-training