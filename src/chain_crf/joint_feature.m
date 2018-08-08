function [psi_feat] = joint_feature(x_sequence, y_sequence, nb_classes, ...
    pairwise_mode, precomputed_pairwise, skip_chain_length)
%% JOINT_FEATURE Compute SSVM joint feature Psi(x,y)
%
% Inputs:
% 
% x_sequence: feat_dim x nb_timesteps
% y_sequence: 1 x nb_timesteps
% nb_classes: number of classes
% pairwise_mode: 'joint': learn nb_classesxnb_classes parameters for the
%                         pairwise term
%                'pre': use precomputed pairwise matrix and tune scaling
%                       parameter lambda_pairwise
% precomputed_pairwise: nb_classes x nb_classes matrix with precomputed
%                       pairwise term
% skip_chain_length: skip chain length

%% Compute joint feature vector corresponding to unary potentials

feat_dim = size(x_sequence, 1);
nb_timesteps = size(x_sequence, 2);

% unary marginals: binary matrix of size T x C (nb)timesteps x nb_classes)
% Each row is an one-hot encoding of the label (has zeroes in all places,
% except for the index corresponding to y_t)
unary_marginals = full(ind2vec(y_sequence, nb_classes))';

% unary_joint_feat_max size: feat_dim x nb_classes
unary_joint_feat_mat = x_sequence * unary_marginals;

% unary_joint_feat_vec size: feat_dim*nb_classes x 1
unary_joint_feat_vec = reshape(unary_joint_feat_mat, ...
    [feat_dim*nb_classes, 1]);

%% Compute joint feature vector corresponding to pairwise potentials

if strcmp(pairwise_mode, 'joint')
    % pairwise_joint_feat_max: matrix of size nb_classes x nb_classes
    pairwise_joint_feat_mat = unary_marginals(1:...
        (end-skip_chain_length), :)'*unary_marginals(1+skip_chain_length:end, :);
    pairwise_joint_feat_vec = reshape(pairwise_joint_feat_mat', ...
        [nb_classes*nb_classes, 1]);
elseif strcmp(pairwise_mode, 'pre')
    % TODO: maybe optimize this
    % pairwise_joint_feat_vec: scalar (1x1)
    pairwise_joint_feat_vec = 0;
    for t = 1 : nb_timesteps - skip_chain_length
        pairwise_joint_feat_vec = pairwise_joint_feat_vec + ...
            precomputed_pairwise(y_sequence(t), ...
            y_sequence(t+skip_chain_length));
    end
else
    error('Not supported pairwise_mode');
end


%% Vertically concatenate them 
psi_feat = [unary_joint_feat_vec; pairwise_joint_feat_vec];

end
