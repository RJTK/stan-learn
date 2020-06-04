functions{
  // TODO: add_diag is a builtin function that flycheck recognizes
  // TODO: but pystan doesn't.  Do I have multiple stanc versions?
  matrix add_to_diag(matrix A, real a){
    int N = dims(A)[1];
    matrix[N, N] B = A;
    for (i in 1:N)
      B[i, i] = A[i, i] + a;
    return B;
  }

  matrix K_cov_exp_quad(vector[] X, real alpha, real rho){
    return add_to_diag(cov_exp_quad(X, alpha, rho), 1e-12);
  }

  matrix L_cov_exp_quad(vector[] X, real alpha, real rho){
    return cholesky_decompose(K_cov_exp_quad(X, alpha, rho));
  }
}
