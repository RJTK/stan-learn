I have taken up this project for the purposes of learning a bit more
about Bayesian statistics and how to program with Stan.  I've bulit up
some simple classes and idioms to hopefully make it fairly
straightforward to write a stan model and then interface with it via
the sklearn API -- this is quite convenient for experimenting with
different datasets.

The first model I implemented is simply a linear regression model with
T-distributed output

```
  y0 ~ cauchy(0, 1);
  nu ~ cauchy(0, 1);
  sigma ~ normal(0, 1);  // half-normal
  lam ~ exponential(1);
  theta ~ normal(0, lam);
  y ~ student_t(nu, y0 + Q * theta, sigma);
```

the input data needs to be scaled to unit variance before feeding it
in.  I obviously do not claim that the following examples are
exemplars of remarkable performance -- they are illustrations.

Here's some examples on the Diabetes dataset.  Parameter (marginal)
posteriors:

![alt tag](https://github.com/RJTK/stan-learn/blob/master/stanlearn/examples/figures/Diabetes_params.png)

Out of sample predictions with uncertainty quantification (Linear Regression)

![alt tag](https://github.com/RJTK/stan-learn/blob/master/stanlearn/examples/figures/Diabetes_pred.png)

Another interesting model is VAR(p) models.  So far, I have developed
an AR(p) model, with a parameterization in terms of the reflection
coefficients (see /RJTK/levinson-durbin-recursion or any book on
signal processing).  This guarantees the stability of every posterior
sample.

![alt tag](https://github.com/RJTK/stan-learn/blob/master/stanlearn/examples/figures/time_series_param_posterior.png)

![alt tag](https://github.com/RJTK/stan-learn/blob/master/stanlearn/examples/figures/time_series_ppc.png)
