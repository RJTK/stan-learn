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
in.

Here's some examples on the Diabetes dataset.  Parameter (marginal)
posteriors:

![alt tag](https://github.com/RJTK/stan-learn/blob/master/figures/Diabetes_params.png)

Out of sample predictions with uncertainty quantification:

![alt tag](https://github.com/RJTK/stan-learn/blob/master/figures/Diabetes_pred.png)

Here's also a regplot from the california housing data

![alt tag](https://github.com/RJTK/stan-learn/blob/master/figures/California-Housing_pred_regplot.png)

More examples are in the `examples/` folder.
