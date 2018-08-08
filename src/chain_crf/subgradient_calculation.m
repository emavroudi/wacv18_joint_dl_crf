function grad_w = subgradient_calculation(psi_feat, psi_feat_hat, reg_c)
%% CRF-SSVM Subgradient Calculation
%
% Inputs:
% psi_feat: vector of size joint_feat_dim (joint feature corresponding to
%           ground truth sequence labels)
% psi_feat_hat: vector of size joint_feat_dim (joint feature corresponding
%               to most violating sequence labels)
% reg_c: regularization weight multiplying max-margin objective

grad_w =  reg_c*(psi_feat_hat - psi_feat);

end