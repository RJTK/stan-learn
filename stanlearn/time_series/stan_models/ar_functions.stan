functions {
  vector step_up_inner(real g, vector b, int k);
  vector step_up(vector g);
  real ar_model_lpdf(vector y, vector y0, vector b, real sigma);
  vector ar_model_rng(vector y, vector y0, vector b, real sigma);
  vector ar_model_forecast(vector y, vector y0, vector b);
  matrix make_toeplitz(vector y, vector y0);
  vector reverse(vector x);

  vector step_up(vector g){
    /*
     * Maps a vector of p reflection coefficients g (|g| < 1) to a
     * stable sequence b of AR coefficients.
     */
    int p = dims(g)[1];  // Model order
    vector[p] b;  // AR Coefficients
    // vector[p] b_cpy;  // Memory

    b[1] = g[1];
    if(p == 1)  // a loop 1:0 is backwards not empty
      return -b;

    for(k in 1:p - 1){
      b[:k + 1] = step_up_inner(g[k + 1], b, k);
    }

    return -b;
  }

  vector step_up_inner(real g, vector b, int k){
    /*
     * A single step of the step_up recursion, this is
     * useful for making use of intermediate results
     */
    vector[k + 1] b_ret;
    for(tau in 1:k)
      b_ret[tau] = b[tau] + g * b[k - tau + 1];
    b_ret[k + 1] = g;
    return b_ret;
  }

  // vector inverse_levinson_durbin(vector g, real sigma){
  //   /* Computes the autocorrelation sequence from the reflection
  //    * coefficients and the noise level.  This can / should be
  //    * combined with the step_up recursion to calculate both
  //    * the autocorrelation and the AR coefficients from (gamma, sigma).
  //    */
  //   int p = dims(g)[1];
  //   real r0;
  //   vector[p] r = rep_vector(0, p);
  //   vector[p + 1] r_full;
  //   vector[p] b;
  //   vector[p] b_cpy;

  //   r0 = sigma^2;
  //   for(tau in 1:p)
  //     r0 /= (1 - g[tau]^2);

  //   for(k in 1:p - 1){
  //     b_cpy = b;
  //     for(tau in 1:k){
  //       b[tau] = b_cpy[tau] + g[k + 1] * b_cpy[k - tau + 1];
  //     }

  //     b[k + 1] = g[k + 1];
  //     for(tau in 1:k + 1)
  //       r[k + 1] += -b[tau] * r[k + 1 - tau];
  //   }
  //   r_full[1] = r0;
  //   r_full[2:] = r;
  //   return r_full;
  // }

  // matrix chol_factor_g(vector g, real sigma){
  //   int p = dims(g)[1];
  //   vector[p + 1] eps;  // Errors at different modelling orders
  //   matrix[p + 1, p + 1] E;  // Diag(eps)
  //   matrix[p + 1, p + 1] B;  // Upper diag matrix of AR coef sequence

  //   eps[p + 1] = sigma^2;
  //   for(tau in 1:p){
  //     eps[p + 1 - tau] = eps[p + 2 - tau] / (1 - g[p + 1 - tau]^2);
  //   }
  //   E = diag_matrix(eps);

  //   B = add_diag(B, 1);  // ones on the diagonal
  //   for(k in 1:p){
  //     B[:k, k + 1] = -step_up()
  //   }
    
  // }

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

  vector reverse(vector x){
    int p = dims(x)[1];
    vector[p] x_rev;
    for (tau in 1:p)
      x_rev[tau] = x[p - tau + 1];
    return x_rev;
  }
}
