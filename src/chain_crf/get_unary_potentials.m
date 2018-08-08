function unary_potentials = get_unary_potentials(w, x_sequence, nb_classes)
%% GET_UNARY_POTENTIALS Compute unary potentials
%
% Inputs
%
% w : vector of size joint_feature_dim
% x_sequence: matrix of size feat_dim x nb_timesteps
%
% Returns:
% unary_potentials: matrix of size nb_classes x nb_timesteps

[feat_dim, nb_timesteps] = size(x_sequence);
unary_weights = reshape(w(1:feat_dim*nb_classes), [feat_dim, nb_classes])';

unary_potentials = unary_weights*x_sequence;

end
