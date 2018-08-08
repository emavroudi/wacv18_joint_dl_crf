function pairwise_potentials = get_pairwise_potentials(w, pairwise_mode, ...
    precomputed_pairwise, nb_classes, feat_dim)
%% GET_PAIRWISE_POTENTIALS
%
% Inputs
% w: weights vector of size joint_feat_dim
% pairwise_mode: 'joint': learn nb_classesxnb_classes parameters for the
%                         pairwise term
%                'pre': use precomputed pairwise matrix and tune scaling
%                       parameter lambda_pairwise
% precomputed_pairwise: nb_classes x nb_classes matrix with precomputed
%                       pairwise term
%
% Output:
% pairwise_potentials: matrix of size nb_classes x nb_classes

if strcmp(pairwise_mode, 'pre')
    % scalar weight multiplying precomputed pairwise matrix
    pairwise_potentials = w(end)*precomputed_pairwise;
elseif strcmp(pairwise_mode, 'joint')
    pairwise_potentials = reshape(w(nb_classes*feat_dim + 1:end), ...
        [nb_classes, nb_classes])';
else
    error('Not supported pairwise_mode');
end
end

