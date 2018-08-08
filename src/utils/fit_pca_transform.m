function [Ud, S] = fit_pca_transform(X, d)
%% FIT_PCA_TRANSFORM Fits PCA on data X
%
% Inputs
% X: centered input data, size: feat_dim x nb_samples
% d: number of principal components
%
% Outputs:
% Ud: Orthonormal basis for the subspace
% S: diagonal matrix with non-zero singular values

[U, S, V] = svd(X, 'econ');
Ud = U(:, 1:d);
S = S(1:d,1:d);
