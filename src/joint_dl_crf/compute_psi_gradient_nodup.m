function [gradPsi_temp, diff_feature_norm, gradPsi_temp_norm] = ...
    compute_psi_gradient_nodup(t, Uhat, half_win, ...
    unary_weights, y_train, y_pred, psi_n, psi_m, window_size, Dhat, psi)
%% COMPUTE_PSI_GRADIENT Computes gradient w.r.t to psi of crf objective
% function depending on psi for one frame.
%
% Inputs:
% t: frame index
% Uhat: sparse codes per frame
% half_win: half window length
% unary_weights: unary weights of size (feat_dim, nb_classes)
% y_train: sequence of ground truth labels
% y_pred: sequence of predicted labels
% psi_n: feat_dim
% psi_m: dictionary size
% window_size: window_size
% Dhat: kinematic features per frame (left and right padded with zeros with
%       length half_win)
% psi: dictionary, matrix of size (feat_dim, dictionary_size)
%
% Outputs:
% gradPsi_temp: gradient w.r.t. dictionary for frame t
% diff_feature_norm: euclidean norm of uw_{\hat{y}_t} - uw_{y_t}
% gradPsi_temp_norm: frobenius norm  of gradient of cost w.r.t dictionary
%                    for frame t

Uhat_temp = Uhat(:, t-half_win:t+half_win);

diff_feature = -unary_weights(:,y_train(t)) + unary_weights(:,y_pred(t));

gradPsi_temp = zeros(psi_n,psi_m);
for j=1:window_size
    
    [diff_psi,activeset] = diffDic_nodup(Dhat(:,t-half_win-1+j),psi,Uhat_temp(:,j),diff_feature);
    gradPsi_temp(:,activeset) = gradPsi_temp(:,activeset) + diff_psi;
end

gradPsi_temp = gradPsi_temp/window_size;
diff_feature_norm = norm(diff_feature);
gradPsi_temp_norm = norm(gradPsi_temp, 'fro');
