# Logging Trainable Parameters (HOWTO)

In order to count and log the number of trainable parameters, we can simply sum
over the size of the leaves of model.params, for instance, right after creating
the model.

@insert_branch_diff howto-logging-parameters