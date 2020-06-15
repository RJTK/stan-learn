/*
 * A mixture of AR(p) models up to a maximum order p_max.
 * Information on ragged datastructures in stan is relevant
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

  simplex[p_max + 1] mu_th;  // Dirichlet mean for theta

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

  vector<lower=0, upper=1>[p_max] mu_beta;  // Mean vector for gamma
  real<lower=0> nu_th; // pseudo-samples for Dirichlet
  real<lower=0> nu_beta;  // pseudo-samples gamma
  vector<lower=-1, upper=1>[p_max] gamma;  // Reflection coefficients

  real<lower=0> sigma;  // noise level
}

transformed parameters {
  vector[n_params] b;  // AR Coefficients

  {int pos = 1;
    // Compute the actual AR coefficients
    for(p in 1:p_max){
      b[pos:pos + model_size[p] - 1] = step_up(gamma[:p]);
      pos += model_size[p];}}}

model {
  vector[p_max] alpha;  // transformed gamma prior params
  vector[p_max] beta;
  vector[p_max + 1] alpha_th;  // transformed Dirichlet prior for theta

  vector[p_max + 1] lpdfs;  // mixture pdfs
  vector[p_max + T] trend;  // trend term

  nu_th ~ inv_gamma(3, 3);
  alpha_th = mu_th * nu_th;

  theta ~ dirichlet(alpha_th);

  // Noise level in the signal
  sigma ~ normal(0, 5);

  // Priors for the reflection coefficients
  mu_beta ~ uniform(0, 1);  // A vector
  nu_beta ~ inv_gamma(3, 3);
  alpha = mu_beta * nu_beta;
  beta = (1 - mu_beta) * nu_beta;

  // trend parameters
  r ~ normal(0, 2);  // The linear time coefficient
  mu ~ normal(0, 2);  // A mean offset

  // The simple trend term
  trend = mu + r * t;

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

generated quantities {
  vector[T] y_ppc;
  vector[T] trend;  // trend term
  int p = categorical_rng(theta);
  int pos;

  trend = mu + r * t[p_max + 1:];
  y_ppc = trend;

  if(p == 1){  // AR(1)
    y_ppc += ar_model_rng(y, y0[:1], b[:1], sigma);
  }
  else if(p == p_max + 1){  // AR(0)
    for(i in 1:T)
      y_ppc[i] += normal_rng(0, sigma);
  }
  else{  // AR(p), p > 1
    pos = 1 + sum(model_size[:p - 1]);
    y_ppc += ar_model_rng(y, y0[:model_size[p]], b[pos:pos + model_size[p] - 1], sigma);
  }
}


// Exception: assign: Rows of left-hand-side (305) and rows of right-hand-side (300) must match in size  (in 'BayesAR_model.stan' at line 171)


// Exception: add: Rows of m1 (305) and rows of m2 (300) must match in size  (in 'BayesAR_model.stan' at line 174)

// Exception: []: accessing element out of range. index 6 out of range; expecting index to be between 1 and 5; index position = 1model_size  (in 'BayesAR_model.stan' at line 174)
