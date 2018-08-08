function unary_weights = get_unary_weights(w, feat_dim, nb_classes)
%% GET_UNARY_WEIGHTS Compute unary weights
%
% Inputs
%
% w : vector of size joint_feature_dim
% feat_dim: feature size
% nb_classes: number of classes
%
% Returns:
% unary_weights: matrix of size feat_dim x nb_classes

unary_weights = reshape(w(1:feat_dim*nb_classes), [feat_dim, nb_classes]);

end
