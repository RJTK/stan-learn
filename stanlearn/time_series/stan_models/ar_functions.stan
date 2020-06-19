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

  matrix make_toeplitz(vector y, vector y0){
    int p = dims(y0)[1];
    int T = dims(y)[1];    
    vector[p + T] y_full;
    matrix[T, p] Y;

    y_full[:p] = y0;
    y_full[p + 1:] = y;

    for(tau in 1:p)
      Y[:, tau] = y_full[tau:tau + T - 1];
    return Y;
  }

  vector ar_model_forecast(vector y, vector y0, vector b){
    int p = dims(b)[1];
    int T = dims(y)[1];
    vector[T] y_hat;
    vector[p + T] y_full;
    vector[p] b_rev = reverse(b);

    y_full[:p] = y0;
    y_full[p + 1:] = y;

    // A Toeplitz type would be a huge boon to this calculation
    // surprisingly though using make_toeplitz, even if it's kept
    // fixed by non-random y0, doesn't seem to help.
    for(t in 1:T){
      y_hat[t] = dot_product(b_rev, y_full[t:t + p - 1]);
    }
    return y_hat;
  }

  real ar_model_lpdf(vector y, vector y0, vector b, real sigma){
    int T = dims(y)[1];
    vector[T] y_hat = ar_model_forecast(y, y0, b);
    return normal_lpdf(y | y_hat, sigma);
  }

  vector ar_model_rng(vector y, vector y0, vector b, real sigma){
    int T = dims(y)[1];
    vector[T] y_hat = ar_model_forecast(y, y0, b);
    vector[T] y_rng;
    for(t in 1:T)
      y_rng[t] = normal_rng(y_hat[t], sigma);
    return y_rng;
  }
}
