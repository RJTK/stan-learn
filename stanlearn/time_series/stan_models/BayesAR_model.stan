/*
 * A mixture of AR(p) models up to a maximum order p_max.
 * Information on ragged datastructures is relevant:
 * https://mc-stan.org/docs/2_23/stan-users-guide/ragged-data-structs-section.html
 *
 */

functions {
  vector step_up(vector g){
    /*
     * Maps a vector of p reflection coefficients g (|g| < 1) to a
     * stable sequence b of AR coefficients.
     */
    int p = dims(g)[1];  // Model order
    vector[p] b;  // AR Coefficients
    vector[p] b_cpy;  // Memory

    b[1] = g[1];
    if(p == 1)
      return -b;

    for(k in 1:p - 1){
        b_cpy = b;
        for(tau in 1:k){
          b[tau] = b_cpy[tau] + g[k + 1] * b_cpy[k - tau + 1];
        }
        b[k + 1] = g[k + 1];
      }

    return -b;
  }

  vector reverse(vector x){
    int p = dims(x)[1];
    vector[p] x_rev;
    for (tau in 1:p)
      x_rev[tau] = x[p - tau + 1];
    return x_rev;
  }

  vector ar_model_forecast(vector y, vector y0, vector b, real sigma){
    int p = dims(b)[1];
    int T = dims(y)[1];
    vector[T] y_hat;
    vector[p] b_rev = reverse(b);

    // Prepend the initial values y0
    vector[p + T] y_full;
    y_full[:p] = y0;
    y_full[p + 1:] = y;

    for(t in 1:T){
      y_hat[t] = dot_product(b_rev, y_full[t:t + p - 1]);
    }
    return y_hat;
  }

  real ar_model_lpdf(vector y, vector y0, vector b, real sigma){
    int T = dims(y)[1];
    vector[T] y_hat = ar_model_forecast(y, y0, b, sigma);
    return normal_lpdf(y | y_hat, sigma);
  }

  vector ar_model_rng(vector y, vector y0, vector b, real sigma){
    int T = dims(y)[1];
    vector[T] y_hat = ar_model_forecast(y, y0, b, sigma);
    vector[T] y_rng;
    for(t in 1:T)
      y_rng[t] = normal_rng(y_hat[t], sigma);
    return y_rng;
  }
}

data {
  int<lower=1> T;  // Number of examples
  int<lower=1> p_max;  // The model order
  vector[T] y;  // the data series y(t)
}

transformed data {
  vector[p_max + T] t;  // "time"
  int n_params = (p_max * (p_max + 1)) / 2;
  int model_size[p_max];
  for(p in 1:p_max)  // For a raged data structure
    model_size[p] = p;

  for(i in -p_max + 1:T)
    t[i + p_max] = i;
  t = t / T;  // Normalize to between [-p/T, 1]
}

parameters {
  real mu;  // Mean value
  real r;  // Linear trend coefficient
  vector[p_max] y0;  // Initial values
  simplex[p_max + 1] theta;  // Mixture parameters

  real<lower=0> nu_beta;  // pseudo-samples gamma
  vector<lower=0, upper=1>[n_params] mu_beta;  // Mean vector for gamma
  vector<lower=-1, upper=1>[n_params] gamma;  // Reflection coefficients

  real<lower=0> sigma;  // noise level
}

transformed parameters {
  vector[n_params] b;  // AR Coefficients

  {int pos = 1;
    // Compute the actual AR coefficients
    for(p in 1:p_max){
      b[pos:pos + model_size[p] - 1] =
        step_up(gamma[pos:pos + model_size[p] - 1]);
      pos += model_size[p];}}}

model {
  vector[n_params] alpha;  // Params for beta prior on g
  vector[n_params] beta;

  // simplex[p_max + 1] psi_th;  // Dirichlet mean for theta
  // real nu_th; // pseudo-samples for Dirichlet

  vector[p_max + 1] lpdfs;  // The mixture pdfs
  vector[p_max + T] trend = mu + r * t;

  // TODO: Sensible prior on theta should prefer simpler models

  // Noise level in the signal
  sigma ~ normal(0, 5);

  // Priors for the reflection coefficients
  mu_beta ~ uniform(0, 1);  // A vector
  nu_beta ~ inv_gamma(3, 3);  // Keep (alpha, beta) > 1 else we get a U shape
  {int pos = 1;
    for(p in 1:p_max){
      alpha[pos:pos + model_size[p] - 1] =
        mu_beta[pos:pos + model_size[p] - 1] * nu_beta;

      beta[pos:pos + model_size[p] - 1] =
        (1 - mu_beta[pos:pos + model_size[p] - 1]) * nu_beta;

      pos = pos + model_size[p];}}

  // trend parameters
  r ~ normal(0, 2);  // The linear time coefficient
  mu ~ normal(0, 2);  // A mean offset

  // Initial values
  y0 ~ normal(trend[:p_max], sigma);

  // Mixture AR(p), including "AR(0)" (pure noise)
  {int pos = 1;
    vector[T] detrend_y = y - trend[p_max + 1:];
    for(p in 1:p_max){
      lpdfs[p] = theta[p] + ar_model_lpdf(detrend_y | y0[:model_size[p]],
                                          b[pos:pos + model_size[p] - 1],
                                          sigma);
      pos += model_size[p];
    }
    lpdfs[p_max + 1] = theta[p_max + 1] + normal_lpdf(detrend_y | 0, sigma);
  }

  target += beta_lpdf(0.5 * (1 + gamma) | alpha, beta);  // in (0, 1)
  target += log_sum_exp(lpdfs);
}

// generated quantities {
//   vector[T] y_ppc;
//   y_ppc = trend[p + 1:] + ar_model_rng(y, y0, b, sigma);
// }
