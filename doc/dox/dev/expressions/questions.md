# Expression Layer Questions

This is just a collection of random musings/questions about the expression
layer.

Why do the engines not obey RAII? *i.e.* why all the init functions?

Should `init_vars`, `init_struct`, and `init_distribution` really shadow their
base class versions?

Why have the override pointer propagate the entire expression stack vs. having
the override occur at the top?

Why is permutation not part of the expression layer? Would decouple much of the
logic.
