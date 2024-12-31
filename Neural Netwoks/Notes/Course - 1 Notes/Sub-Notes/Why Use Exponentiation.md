Using exponentiation provides a compact way to represent the likelihood for both classes without needing separate conditions. The exponents effectively "switch" the contribution based on the value of \($y_i$\):

- If \($y_i = 1$\):
  - \($a_i^{y_i} = a_i^1 = a_i$\) contributes to the likelihood.
  - \($(1 - a_i)^{(1 - y_i)} = (1 - a_i)^{0} = 1$\) does not contribute.

- If \($y_i = 0$\):
  - \($a_i^{y_i} = a_i^{0} = 1$\) does not contribute.
  - \($(1 - a_i)^{(1 - y_i)} = (1 - a_i)^{1} = 1 - a_i$\) contributes to the likelihood.

### Summary

Thus, the use of powers in the likelihood expression \($a_i^{y_i} (1 - a_i)^{(1 - y_i)}$\) allows for a unified formula that succinctly captures the contribution of predicted probabilities to the likelihood of observing the true label, depending on whether that label is 0 or 1. This compact form is crucial for mathematical derivations in maximum likelihood estimation and optimization.