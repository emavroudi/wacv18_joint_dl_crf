function [diffpsi,activeset_mod] = diffDic(x,psi,u,diffU)

activeset = find(u~=0);
activeset_mod = mod(activeset,size(psi,2));
activeset_mod(activeset_mod==0) = size(psi,2);
psi_active = psi(:,activeset_mod);
A = inv(psi_active'*psi_active);
diffU = diffU(activeset);

diffpsi = (x-psi_active*u(activeset))*(A*diffU)'-psi_active*A'*diffU*u(activeset)';

