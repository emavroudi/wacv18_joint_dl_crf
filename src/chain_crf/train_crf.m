function [w, optimization_log] = train_crf(training_data, training_labels, ...
    nb_epochs, reg_c, learning_rate, batch_size, decay_period, decay_factor,...
    decay_t0, decay_exponent, use_momentum, momentum, optimizer, ...
    pairwise_mode, precomputed_pairwise, w_init, ...
    validation_data, validation_labels, skip_chain_length, nb_classes, ...
    random_seed)
%% TRAIN_CRF Trains CRF using a max-margin approach (SSVM)
%
% Inputs
%
% training_data: cell with nb_training_samples elements. Each one is of
%                size feat_dim x timesteps
% training_labels: cell with nb_training_samples elements. Each one is
%                  vector with length=timesteps, each element of which is a
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
% validation_data: cell with nb_validation_samples elements. Each one is of
%                  size feat_dim x timesteps
% validation_labels: array with nb_validation_samples elements. Each one is a
%                  label from 1 to nb_classes
% skip_chain_length: skip chain length (used in pairwise term)
% nb_classes: number of action classes
% random_seed: random seed
%
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
% Date: 4/21/2017

rng(random_seed);
w = w_init;

training_loss = zeros(1, nb_epochs);
training_unreg_loss = zeros(1, nb_epochs);
training_accuracy = zeros(1, nb_epochs);
gradw_norm= zeros(1, nb_epochs);
w_diff_norm = zeros(1, nb_epochs);
validation_loss = zeros(1, nb_epochs);
validation_unreg_loss = zeros(1, nb_epochs);
validation_accuracy = zeros(1, nb_epochs);

%% Stochastic Gradient Descent
grad_w_old = [];
batch_cnt = 0;

for epoch = 1 : nb_epochs
    tic;

    fprintf('Epoch: %d\n', epoch);

    % Shuffle samples
    nb_training_samples = length(training_data);
    ind_rnd = randperm(nb_training_samples);

    gradw_norm_per_batch = [];
    training_loss_per_batch = [];
    training_unreg_loss_per_batch = [];
    training_accuracy_per_batch = [];
    w_diff_norm_per_batch = [];


    for batch = 1 : nb_training_samples
        % fprintf('\t Batch: %d\n', batch);
        % Get sequences of training features and training labels
        % x_sequence: feat_dim x timesteps
        % y_sequence: 1 x timesteps
        %x_sequence = training_data{batch};
        %y_sequence = training_labels{batch};
        x_sequence = training_data{ind_rnd(batch)};
        y_sequence = training_labels{ind_rnd(batch)};

        % TODO: only compute this for each x_sequence, y_sequence pair
        % once
        % Compute joint feature Psi(x,y)
        psi_feat = joint_feature(x_sequence, y_sequence, nb_classes,...
            pairwise_mode, precomputed_pairwise, skip_chain_length);
        if isempty(w)
            if strcmp(pairwise_mode, 'pre')
                %w = randn(size(psi_feat));
                w = zeros(size(psi_feat));
            else
                w = zeros(size(psi_feat));
            end
        end

        % Solve for most violated constraint by loss augmented inference
        feat_dim = size(x_sequence, 1);
        unary_potentials = get_unary_potentials(w, x_sequence, nb_classes);
        pairwise_potentials = get_pairwise_potentials(w, pairwise_mode, ...
            precomputed_pairwise, nb_classes, feat_dim);
        y_sequence_hat = loss_augmented_inference(unary_potentials, ...
            pairwise_potentials, y_sequence, skip_chain_length);
        % y_sequence_hat(1001:1110)
        % Compute joint feature Psi(x,y_hat)
        psi_feat_hat = joint_feature(x_sequence, y_sequence_hat, ...
            nb_classes, pairwise_mode, precomputed_pairwise, ...
            skip_chain_length);
        % Compute gradient of CRF parameters
        grad_w = subgradient_calculation(psi_feat, psi_feat_hat, reg_c);
        grad_w = grad_w + w;

        if use_momentum
            if isempty(grad_w_old)
                grad_w_old = zeros(size(psi_feat));
            end
            grad_w_old = (1 - momentum)*grad_w + momentum*grad_w_old;
            grad_w = grad_w_old;
        end

        % Update Learning Rate
        % Update the learning rate
        if strcmp(optimizer, 'const_lr')
            lr = learning_rate
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

        % Update CRF weights
        w_prev = w;
        w = w - lr*grad_w;
        if strcmp(pairwise_mode, 'pre')
            w(end) = max(w(end), 0);
            %fprintf('Batch: %d, w_end: %f, grad_w_end: %f\n', batch, w(end), grad_w(end));
        end
        % w(1:10)
        % Predict most probable sequence of labels
        y_pred = inference(unary_potentials, ...
            pairwise_potentials, skip_chain_length);
        if batch == -1 % batch == 36 || batch == 37 || batch == 38 || batch == 39
            y_sequence_hat
            unary_potentials
            max(unary_potentials(:))
            mean(unary_potentials(:))
            min(unary_potentials(:))
            max(pairwise_potentials(:))
            y_pred
        end

        training_accuracy_per_batch = [training_accuracy_per_batch, ...
            mean(y_pred==y_sequence)*100];
        % TODO: training accuracy
        % Log loss, unregularized loss, w diff, gradient norm,
        % training accuracy
        batch_training_unreg_loss = max_margin_loss(w, ...
            psi_feat, psi_feat_hat, y_sequence, y_sequence_hat, reg_c);
        batch_training_loss = (1/2)*norm(w) + ...
            batch_training_unreg_loss;
        batch_gradw_norm = norm(grad_w);
        batch_w_diff_norm = norm(w - w_prev);

        training_loss_per_batch = [training_loss_per_batch, ...
            batch_training_loss];
        training_unreg_loss_per_batch = [training_unreg_loss_per_batch, ...
            batch_training_unreg_loss];
        gradw_norm_per_batch = [gradw_norm_per_batch, batch_gradw_norm];
        w_diff_norm_per_batch = [w_diff_norm_per_batch, batch_w_diff_norm];
        batch_cnt = batch_cnt + 1;
    end
    %training_loss_per_batch
    %training_accuracy_per_batch
    % Log optimization/performance metrics on epoch end
    training_unreg_loss(epoch) = mean(training_unreg_loss_per_batch);
    training_loss(epoch) = mean(training_loss_per_batch);
    w_diff_norm(epoch) = norm(w - w_prev);
    gradw_norm(epoch) = mean(gradw_norm_per_batch);
    training_accuracy(epoch) = mean(training_accuracy_per_batch);
    fprintf('Epoch %d, unreg_loss: %f, loss: %f, acc: %f, gradw_norm: %f, w_diff_norm: %f, w_end %f\n', ...
        epoch, training_unreg_loss(epoch), training_loss(epoch), training_accuracy(epoch), ...
        w_diff_norm(epoch), gradw_norm(epoch), w(end));

    % Compute loss in validation set (mean loss over all validation
    % sequences) and compute average frame level accuracy in validation set
    validation_loss_per_sample = [];
    validation_unreg_loss_per_sample = [];
    validation_acc_per_sample = [];
    validation_y_true = {};
    validation_y_pred = {};
    validation_acc_per_seq = [];

    for i = 1 : length(validation_data)
        x_sequence = validation_data{i};
        y_sequence = validation_labels{i};

        validation_y_true{i} = y_sequence;

        % Compute joint feature Psi(x,y)
        psi_feat = joint_feature(x_sequence, y_sequence, nb_classes,...
            pairwise_mode, precomputed_pairwise, skip_chain_length);

        % Solve for most violated constraint by loss augmented inference
        feat_dim = size(x_sequence, 1);
        unary_potentials = get_unary_potentials(w, x_sequence, nb_classes);
        pairwise_potentials = get_pairwise_potentials(w, pairwise_mode, ...
            precomputed_pairwise, nb_classes, feat_dim);
        y_sequence_hat = loss_augmented_inference(unary_potentials, ...
            pairwise_potentials, y_sequence, skip_chain_length);

        % Compute joint feature Psi(x,y_hat)
        psi_feat_hat = joint_feature(x_sequence, y_sequence_hat, ...
            nb_classes, pairwise_mode, precomputed_pairwise, ...
            skip_chain_length);

        % Predict most probable sequence of labels
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
optimization_log.validation_loss = validation_loss;
optimization_log.validation_unreg_loss = validation_unreg_loss;
optimization_log.validation_accuracy = validation_accuracy;
optimization_log.validation_y_true = validation_y_true;
optimization_log.validation_y_pred = validation_y_pred;
optimization_log.validation_acc_per_seq = validation_acc_per_seq;
optimization_log.pairwise_potentials = pairwise_potentials;
