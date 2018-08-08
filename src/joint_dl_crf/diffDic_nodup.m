function [diffpsi,activeset] = diffDic_nodup(x,psi,u,diffU)

% u: vector of sparse codes with size: (dict_size, 1)

% Find indices of non-zero sparse codes
activeset = find(u~=0);
% Get active columns of psi
psi_active = psi(:,activeset);
% Compute A
A = inv(psi_active'*psi_active);
% Get active entries of gradient w.r.t to z_i, size (dict_size, 1)
diffU = diffU(activeset);

diffpsi = (x-psi_active*u(activeset))*(A*diffU)'-psi_active*A'*diffU*u(activeset)';

