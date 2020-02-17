# Polyak Averaging (HOWTO)

With Polyak Averaging, the training loop keeps track of additional parameters, 
which are the exponential moving average of the parameters over the course of 
training. Then, making predictions with these EMA parameters typically leads to 
better predictions.

@insert_branch_diff howto-polyak-averaging