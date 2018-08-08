function [features,Uhats] = sumpool_dup(Uhat)

Uhat_pos = +full(Uhat>0);
Uhat_neg = +full(Uhat<0);

% duplicating the sparse codes to positive and negative components
Uhat_pos = full(Uhat_pos.*Uhat);
Uhat_neg = full(Uhat_neg.*Uhat); % negative components keep their sign 


Uhats = [Uhat_pos;Uhat_neg];
features = sum(Uhats,2);
