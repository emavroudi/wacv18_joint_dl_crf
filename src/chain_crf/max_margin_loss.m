function loss = max_margin_loss(w, psi_feat, psi_feat_hat, y_sequence, ...
    y_sequence_hat, reg_c)
%% MAX_MARGIN_LOSS 
%
% Input
% w: weight vector of size joint_feat_dim 
% psi_feat: vector of size joint_feat_dim (joint feature corresponding to
%           ground truth sequence labels)
% psi_feat_hat: vector of size joint_feat_dim (joint feature corresponding
%               to most violating sequence labels)
% y_sequence: ground truth sequence of labels (vector of length
%             nb_timesteps)
% y_sequence_hat: most violating sequence of labels/most violated 
%                 constraint (result of loss_augmented_inference)
% reg_c: regularization weight multiplying objective
loss = reg_c*(hamming_loss(y_sequence, y_sequence_hat) + ...
    dot(w, psi_feat_hat) - dot(w, psi_feat));
% fprintf('Hamming loss %d\n', hamming_loss(y_sequence, y_sequence_hat));
end
