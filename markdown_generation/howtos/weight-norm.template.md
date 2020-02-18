# Weight Normalization (HOWTO)

Adds weight normalization to an optimizer def. Weight vectors are decomposed as
`w = g * v/||v||_2`, for scalar scale parameter `g`, and raw weight vector `v`.
The original optimizer is then applied to the `(g, v)` parameterization and the
updated parameters are transformed back to w-space, i.e. 
`w,state --> (g,v) --(original optimizer)--> (g',v') --> w',state'`.

@insert_branch_diff howto-weight-norm
