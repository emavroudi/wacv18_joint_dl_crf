function [Y] = pca_transform(X, U, S, whiten, epsilon)
%% PCA_TRANSFORM Projects points in X to orthonormal basis
% U for dimensionality reduction
%
% Inputs
% X: centered data matrix, size: feat_dim x nb_samples
% U: Orthonormal basis for the subspace, size: feat_dim x nb_components
% S: diagonal matrix with singular values
%
% Outputs:
% Y: projected data, size: nb_components x nb_samples

Y = U'*X;

if whiten
     Y = diag(1./sqrt(diag(S) + epsilon)) * U' * X;
end