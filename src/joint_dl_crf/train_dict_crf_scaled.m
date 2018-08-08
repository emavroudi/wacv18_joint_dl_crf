function [w, psi, optimization_log] = train_dict_crf_scaled( ...
    training_data, training_labels, ...
    validation_data, validation_labels, nb_epochs, reg_c, learning_rate, ...
    batch_size, decay_period, decay_factor,...
    decay_t0, decay_exponent, use_momentum, momentum, optimizer, ...
    pairwise_mode, precomputed_pairwise, w_init, ...
    skip_chain_length, nb_classes, ...
    psi_init, window_size, mex_lasso_param, random_seed, ...
    gradpsi_scaling_factor)
%% TRAIN_DICT_CRF_SCALED Jointly trains Dictionary and CRF using
%  a max-margin approach (SSVM) (with dict gradient scaling)
%
% Inputs
%
% training_data: cell with nb_training_samples elements. Each one is of
%                size feat_dim x timesteps
% training_labels: cell with nb_training_samples elements. Each one is
%                  vector with length=timesteps, each element of which is a
%                  label from 1 to nb_classes
% validation_data: cell with nb_validation_samples elements. Each one is of
%                  size feat_dim x timesteps
% validation_labels: array with nb_validation_samples elements. Each one is a
%                  label from 1 to nb_classes
% nb_epochs: number of epochs
% reg_c: weight for regularization (multiplies max-margin objective)
% learning_rate: (initial) learning rate
% batch_size: batch size, for now we support only batch_size=1
% decay_period: decay learning rate every decay_period epochs (use -1 if no
%               decay)
% decay_factor: decay learning rate by decay factor (use -1 if no decay)
% pairwise_mode: 'joint': learn nb_classesxnb_classes parameters for the
%                         pairwise term
%                'pre': use precomputed pairwise matrix and tune scaling
%                       parameter lambda_pairwise
% precomputed_pairwise: nb_classes x nb_classes matrix with precomputed
%                       pairwise term
% w_init: initialization of unary and pairwise weights. vector of size
% (size_joint_feature) (e.g. nb_classes*feat_dim + nb_classes*nb_classes)
% skip_chain_length: skip chain length (used in pairwise term)
% nb_classes: number of action classes
% psi_init: initialization of dictionary,
%           matrix of size (feat_dim, dict_size)
% window_size: window size
% mex_lasso_param: param.K = psi_m;
%                  param.lambda=lambda_lasso;
%                  param.numThreads=-1; % number of threads
%                  param.verbose=false;
%                  param.iter=1000;
% random_seed: random seed
% gradpsi_scaling_factor: scaling factor for gradient with respect to psi
% Outputs:
%
% w: CRF weights, vector of size joint_feature_size
% optimization_log: struct with fields:
%   training_loss
%   training_unreg_loss
%   training_accuracy
%   validation_loss
%   validation_unreg_loss
%   validation_accuracy
%   gradw_norm
%   w_diff_norm
%
% See also: joint_feature, loss_augmented_inference, inference,
% subgradient_calculation, max_margin_loss
%
% Author: Efi Mavroudi (efi.mavroudi@gmail.com)
% Date: 5/21/2017

setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')

rng(random_seed);

w = w_init;
psi = psi_init;
[psi_n, psi_m] = size(psi);
half_win = (window_size-1)/2;

training_loss = zeros(1, nb_epochs);
training_unreg_loss = zeros(1, nb_epochs);
training_accuracy = zeros(1, nb_epochs);
gradw_norm= zeros(1, nb_epochs);
unreg_gradw_norm_before_mom = zeros(1, nb_epochs);
gradw_norm_before_mom = zeros(1, nb_epochs);
gradpsi_norm_before_mom = zeros(1, nb_epochs);
diff_feature_norm = zeros(1, nb_epochs);
gradPsi_temp_norm = zeros(1, nb_epochs);
w_diff_norm = zeros(1, nb_epochs);
gradpsi_norm= zeros(1, nb_epochs);
psi_diff_norm = zeros(1, nb_epochs);
validation_loss = zeros(1, nb_epochs);
validation_unreg_loss = zeros(1, nb_epochs);
validation_accuracy = zeros(1, nb_epochs);

%% Stochastic Gradient Descent
grad_w_old = [];
grad_psi_old = [];
batch_cnt = 0;


for epoch = 1 : nb_epochs
    tic;

    fprintf('Epoch: %d\n', epoch);

    % Shuffle samples
    nb_training_samples = length(training_data);
    ind_rnd = randperm(nb_training_samples);

    gradw_norm_per_batch = [];
    unreg_gradw_norm_before_mom_per_batch = [];
    gradw_norm_before_mom_per_batch = [];
    gradpsi_norm_per_batch = [];
    gradpsi_norm_before_mom_per_batch = [];
    diff_feature_norm_per_batch = [];
    gradPsi_temp_norm_per_batch = [];
    training_loss_per_batch = [];
    training_unreg_loss_per_batch = [];
    training_accuracy_per_batch = [];
    w_diff_norm_per_batch = [];
    psi_diff_norm_per_batch = [];


    for batch = 1 : nb_training_samples
        % fprintf('\t Batch: %d\n', batch);
        % Get sequences of training features and training labels
        % x_sequence: feat_dim x timesteps
        % y_sequence: 1 x timesteps
        sample_ind = ind_rnd(batch);
        Dhat = training_data{sample_ind};
        %% Compute sparse codes
        Uhat = mexLasso(Dhat, psi, mex_lasso_param);
        % Compute sparse features
        x_sequence = histogram_sparse_codes(Uhat, window_size);
        y_sequence = training_labels{sample_ind};

        % TODO: only compute this for each x_sequence, y_sequence pair
        % once
        %% Compute joint feature Psi(x,y)
        psi_feat = joint_feature(x_sequence, y_sequence, nb_classes,...
            pairwise_mode, precomputed_pairwise, skip_chain_length);
        if isempty(w)
            w = zeros(size(psi_feat));
        end

        %% Solve for most violated constraint by loss augmented inference
        feat_dim = size(x_sequence, 1);
        unary_potentials = get_unary_potentials(w, x_sequence, nb_classes);
        pairwise_potentials = get_pairwise_potentials(w, pairwise_mode, ...
            precomputed_pairwise, nb_classes, feat_dim);
        y_sequence_hat = loss_augmented_inference(unary_potentials, ...
            pairwise_potentials, y_sequence, skip_chain_length);
        % y_sequence_hat(1001:1110)
        %% Compute joint feature Psi(x,y_hat)
        psi_feat_hat = joint_feature(x_sequence, y_sequence_hat, ...
            nb_classes, pairwise_mode, precomputed_pairwise, ...
            skip_chain_length);

        %% Compute training loss/accuracy until now
        % Predict most probable sequence of labels
        y_pred = inference(unary_potentials, ...
            pairwise_potentials, skip_chain_length);

        training_accuracy_per_batch = [training_accuracy_per_batch, ...
            mean(y_pred==y_sequence)*100];
        % TODO: training accuracy
        % Log loss, unregularized loss, w diff, gradient norm,
        % training accuracy
        batch_training_unreg_loss = max_margin_loss(w, ...
            psi_feat, psi_feat_hat, y_sequence, y_sequence_hat, reg_c);
        batch_training_loss = (1/2)*norm(w) + ...
            batch_training_unreg_loss;

        training_loss_per_batch = [training_loss_per_batch, ...
            batch_training_loss];
        training_unreg_loss_per_batch = [training_unreg_loss_per_batch, ...
            batch_training_unreg_loss];

        %% Compute gradient of CRF parameters
        grad_w = subgradient_calculation(psi_feat, psi_feat_hat, reg_c);
        unreg_gradw_norm_before_mom_batch = norm(grad_w);
        grad_w = grad_w + w;
        gradw_norm_before_mom_batch = norm(grad_w);

        if use_momentum
            if isempty(grad_w_old)
                grad_w_old = zeros(size(psi_feat));
            end
            grad_w_old = (1 - momentum)*grad_w + momentum*grad_w_old;
            grad_w = grad_w_old;
        end

        %% Compute gradient of dictionary and new features
        nb_timesteps = size(x_sequence, 2);
        sum_tempPsi = zeros(psi_n,psi_m);
        unary_weights = get_unary_weights(w, feat_dim, nb_classes);

        disp(['Computing gradient w.r.t. psi across ', ...
            num2str(nb_timesteps), ' training examples..']);
        % Pad Uhat, Dhat, y_sequence, y_sequence_hat with zeros
        % TODO: check if needed
        Uhat_pad = zeros(size(Uhat,1),half_win);
        Dhat_pad = zeros(size(Dhat,1),half_win);
        y_pad = zeros(1, half_win);
        Uhat_padded = [Uhat_pad Uhat Uhat_pad];
        Dhat_padded = [Dhat_pad Dhat Dhat_pad];
        y_sequence_padded = [y_pad y_sequence y_pad];
        y_sequence_hat_padded = [y_pad y_sequence_hat y_pad];

        %tic;
        [psi_active_per_frame, A_per_frame, activeset_per_frame, ...
            activeset_mod_per_frame] = precompute_psi_active(...
            Uhat_padded, psi);
        %fprintf('Precomputed psi active, elapsed Time = %f\n', toc)

        %tic;
        diff_feature_norm_sum = 0;
        gradPsi_temp_norm_sum = 0;
        parfor t_par = (half_win + 1) : (nb_timesteps + half_win)
            [gradPsi_temp, diff_feature_norm, gradPsi_temp_norm] = ...
                compute_psi_gradient_eff(t_par, ...
                Uhat_padded, half_win, unary_weights, y_sequence_padded, ...
                y_sequence_hat_padded, ...
                psi_n, psi_m, window_size, Dhat_padded, psi, ...
                psi_active_per_frame, A_per_frame, activeset_per_frame, ...
                activeset_mod_per_frame);
            %[gradPsi_temp, diff_feature_norm, gradPsi_temp_norm] = ...
            %    compute_psi_gradient(t_par, ...
            %    Uhat_padded, half_win, unary_weights, y_sequence_padded, ...
            %    y_sequence_hat_padded, ...
            %    psi_n, psi_m, window_size, Dhat_padded, psi);

            sum_tempPsi = sum_tempPsi + gradPsi_temp;

            diff_feature_norm_sum = diff_feature_norm_sum + diff_feature_norm;
            gradPsi_temp_norm_sum = gradPsi_temp_norm_sum + gradPsi_temp_norm;
        end

        %fprintf('Computed gradient, elapsed Time = %f\n', toc);
        diff_feature_norm_batch = diff_feature_norm_sum / nb_timesteps;
        gradPsi_temp_norm_batch = gradPsi_temp_norm_sum / nb_timesteps;

        % grad_psi = reg_c * sum_tempPsi/nb_timesteps;
        grad_psi = reg_c * sum_tempPsi;
        gradpsi_norm_before_mom_batch = norm(grad_psi, 'fro');
        if use_momentum
            if isempty(grad_psi_old)
                grad_psi_old = zeros(size(psi));
            end
            grad_psi_old = (1 - momentum)*grad_psi + momentum*grad_psi_old;
            grad_psi = grad_psi_old;
        end

        %% Update Learning Rate
        % Update the learning rate
        if strcmp(optimizer, 'const_lr')
            lr = learning_rate;
        elseif strcmp(optimizer, 'block_decay')
            if decay_period ~= -1
                lr = decay_factor^(epoch/decay_period) * learning_rate;
            else
                error('Invalid decay_period')
            end
        elseif strcmp(optimizer, 'batch_decay')
            lr = learning_rate / (batch_cnt + decay_t0)^decay_exponent;
        else
            error('Invalid optimizer');
        end

        %% Update CRF weights
        w_prev = w;
        w = w - lr*grad_w;
        % w(1:10)
        %% Update dictionary and sparse features
        psi_prev = psi;
        psi = psi - gradpsi_scaling_factor*lr*grad_psi;
        psi = normc(psi);

        %% Compute gradient norms
        batch_gradw_norm = norm(grad_w);
        batch_w_diff_norm = norm(w - w_prev);
        batch_gradpsi_norm = norm(grad_psi, 'fro');
        batch_psi_diff_norm = norm(psi - psi_prev, 'fro');
        gradw_norm_per_batch = [gradw_norm_per_batch, batch_gradw_norm];
        unreg_gradw_norm_before_mom_per_batch = [unreg_gradw_norm_before_mom_per_batch, ...
            unreg_gradw_norm_before_mom_batch];
        gradw_norm_before_mom_per_batch = [gradw_norm_before_mom_per_batch, ...
            gradw_norm_before_mom_batch];
        w_diff_norm_per_batch = [w_diff_norm_per_batch, batch_w_diff_norm];
        gradpsi_norm_per_batch = [gradpsi_norm_per_batch, ...
            batch_gradpsi_norm];
        gradpsi_norm_before_mom_per_batch = [gradpsi_norm_before_mom_per_batch, ...
            gradpsi_norm_before_mom_batch];
        diff_feature_norm_per_batch = [diff_feature_norm_per_batch diff_feature_norm_batch];
        gradPsi_temp_norm_per_batch = [gradPsi_temp_norm_per_batch gradPsi_temp_norm_batch];
        psi_diff_norm_per_batch = [psi_diff_norm_per_batch, ...
            batch_psi_diff_norm];


        batch_cnt = batch_cnt + 1;
    end

    %% Log optimization/performance metrics on epoch end
    training_unreg_loss(epoch) = mean(training_unreg_loss_per_batch);
    training_loss(epoch) = mean(training_loss_per_batch);
    w_diff_norm(epoch) = norm(w - w_prev);
    psi_diff_norm(epoch) = norm(psi - psi_prev);
    gradw_norm(epoch) = mean(gradw_norm_per_batch);
    unreg_gradw_norm_before_mom(epoch) = mean(unreg_gradw_norm_before_mom_per_batch);
    gradw_norm_before_mom(epoch) = mean(gradw_norm_before_mom_per_batch);
    gradpsi_norm(epoch) = mean(gradpsi_norm_per_batch);
    gradpsi_norm_before_mom(epoch) = mean(gradpsi_norm_before_mom_per_batch);
    diff_feature_norm(epoch) = mean(diff_feature_norm_per_batch);
    gradPsi_temp_norm(epoch) = mean(gradPsi_temp_norm_per_batch);
    training_accuracy(epoch) = mean(training_accuracy_per_batch);
    fprintf(['Epoch %d, unreg_loss: %f, loss: %f, acc: %f, gradw_norm: %f,', ...
        'unreg_gradw_norm_before_mom: %f, gradw_norm_before_mom: %f,', ...
        'w_diff_norm: %f, gradpsi_norm: %f, gradpsi_norm_before_mom: %f,', ...
        'diff_feature_norm: %f, gradPsi_temp_norm: %f, psi_diff_norm: %f, w_end: %f\n'], ...
        epoch, training_unreg_loss(epoch), training_loss(epoch), ...
        training_accuracy(epoch), gradw_norm(epoch), ...
        unreg_gradw_norm_before_mom(epoch), gradw_norm_before_mom(epoch), ...
        w_diff_norm(epoch), gradpsi_norm(epoch), ...
        gradpsi_norm_before_mom(epoch), ...
        diff_feature_norm(epoch), gradPsi_temp_norm(epoch), ...
        psi_diff_norm(epoch), w(end));

    % Compute loss in validation set (mean loss over all validation
    % sequences) and compute average frame level accuracy in validation set
    validation_loss_per_sample = [];
    validation_unreg_loss_per_sample = [];
    validation_acc_per_sample = [];
    validation_y_true = {};
    validation_y_pred = {};
    validation_acc_per_seq = [];

    for i = 1 : length(validation_data)
        Dhat = validation_data{i};
        %% Compute sparse codes
        Uhat = mexLasso(Dhat, psi, mex_lasso_param);
        % Compute sparse features
        x_sequence = histogram_sparse_codes(Uhat, window_size);
        y_sequence = validation_labels{i};

        validation_y_true{i} = y_sequence;

        %% Compute joint feature Psi(x,y)
        psi_feat = joint_feature(x_sequence, y_sequence, nb_classes,...
            pairwise_mode, precomputed_pairwise, skip_chain_length);

        %% Solve for most violated constraint by loss augmented inference
        feat_dim = size(x_sequence, 1);
        unary_potentials = get_unary_potentials(w, x_sequence, nb_classes);
        pairwise_potentials = get_pairwise_potentials(w, pairwise_mode, ...
            precomputed_pairwise, nb_classes, feat_dim);
        y_sequence_hat = loss_augmented_inference(unary_potentials, ...
            pairwise_potentials, y_sequence, skip_chain_length);

        %% Compute joint feature Psi(x,y_hat)
        psi_feat_hat = joint_feature(x_sequence, y_sequence_hat, ...
            nb_classes, pairwise_mode, precomputed_pairwise, ...
            skip_chain_length);

        %% Predict most probable sequence of labels
        y_pred = inference(unary_potentials, ...
            pairwise_potentials, skip_chain_length);

        validation_y_pred{i} = y_pred;

        validation_sample_unreg_loss = max_margin_loss(w, ...
            psi_feat, psi_feat_hat, y_sequence, y_sequence_hat, reg_c);
        validation_sample_loss = (1/2)*norm(w) + ...
            validation_sample_unreg_loss;
        validation_sample_acc = mean(y_pred==y_sequence)*100;

        validation_acc_per_seq = [validation_acc_per_seq validation_sample_acc];

        validation_unreg_loss_per_sample = [...
            validation_unreg_loss_per_sample, ...
            validation_sample_unreg_loss];
        validation_loss_per_sample = [validation_loss_per_sample, ...
            validation_sample_loss];
        validation_acc_per_sample = [validation_acc_per_sample, ...
            validation_sample_acc];
    end
    validation_unreg_loss(epoch) = mean(validation_unreg_loss_per_sample);
    validation_loss(epoch)= mean(validation_loss_per_sample);
    validation_accuracy(epoch) = mean(validation_acc_per_sample);
    fprintf('Epoch %d, val_unreg_loss: %f, val_loss: %f, val_acc: %f\n', ...
        epoch, validation_unreg_loss(epoch), validation_loss(epoch), ...
        validation_accuracy(epoch));
    fprintf('Epoch %d, elapsed Time = %f\n', epoch, toc);
end


optimization_log.training_unreg_loss = training_unreg_loss;
optimization_log.training_loss = training_loss;
optimization_log.training_accuracy = training_accuracy;
optimization_log.w_diff_norm = w_diff_norm;
optimization_log.gradw_norm = gradw_norm;
optimization_log.psi_diff_norm = psi_diff_norm;
optimization_log.gradpsi_norm = gradpsi_norm;
optimization_log.validation_loss = validation_loss;
optimization_log.validation_unreg_loss = validation_unreg_loss;
optimization_log.validation_accuracy = validation_accuracy;
optimization_log.validation_y_true = validation_y_true;
optimization_log.validation_y_pred = validation_y_pred;
optimization_log.validation_acc_per_seq = validation_acc_per_seq;
optimization_log.pairwise_potentials = pairwise_potentials;
