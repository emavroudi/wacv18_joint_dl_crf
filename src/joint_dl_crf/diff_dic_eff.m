function diffpsi = diff_dic_eff(x, u, diffU, precomputed_psi_active, ...
   precomputed_A, precomputed_activeset)

% Get active entries of gradient w.r.t to z_i, size (2*dict_size, 1)
diffU = diffU(precomputed_activeset);

diffpsi = (x-precomputed_psi_active*u(precomputed_activeset))*(precomputed_A*diffU)'-precomputed_psi_active*precomputed_A'*diffU*u(...
    precomputed_activeset)';
