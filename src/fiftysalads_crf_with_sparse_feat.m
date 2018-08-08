function fiftysalads_crf_with_sparse_feat(output_dir, output_csv_file, ...
    params)
%% CRF_WITH_SPARSE_FEAT: Learning of Conditional Random Field on top
% of sparse codes histogrames using Stochastic Gradient Descent

%% Parameters
granularity = params.granularity;
nb_epochs = params.nb_epochs;
reg_c = params.reg_c;
learning_rate = params.learning_rate;
batch_size = params.batch_size;
decay_period = params.decay_period;
decay_factor = params.decay_factor;
decay_t0 = params.decay_t0;
decay_exponent = params.decay_exponent;
use_momentum = params.use_momentum;
momentum = params.momentum;
optimizer = params.optimizer;
pairwise_mode = params.pairwise_mode;
skip_chain_length = params.skip_chain_length;
normalization = params.normalization;
window_size = params.window_size;
lambda_lasso = params.lambda_lasso;
dict_size = params.dict_size;
use_pca = params.use_pca;
nb_components = params.nb_components;
data_dir = params.data_dir;
feat_type = params.feat_type;
random_seed = params.random_seed;
split = params.split;
final_split = params.final_split;

%% Setup SPAMS
setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')

%% Seed pseudorandom number generator
rng(random_seed);

%% mexLasso parameters
% number of dictionary atoms
mex_lasso_param.K=dict_size;
mex_lasso_param.lambda=lambda_lasso;
% number of threads
mex_lasso_param.numThreads=-1;
mex_lasso_param.verbose=false;
mex_lasso_param.iter=1000;

%% Load standardized data
[training_sequences, testing_sequences, split_str, classes] = ...
    load_fiftysalads_data_lea(data_dir, granularity, split, 1);
nb_classes = length(classes);

format_str = ['sparse_z_e_%d_c_%.3f_lr_%.4f', ...
    '_bs_%d_dp_%d_df_%.2f_dt0_%.2f_dexp_%.2f_um_%d_mom_%.1f_opt_%s_pw_%s', ...
    '_skip_%d_norm_%s', ...
    '_win_%d_lambda_%.2f_dict_%d_pca_%d_n_%d', ...
    '_rs_%d_feat_%s'];
results_dir_name = sprintf(format_str, nb_epochs, reg_c, learning_rate, ...
    batch_size, decay_period, decay_factor, decay_t0, ...
    decay_exponent, use_momentum, momentum, optimizer, ...
    pairwise_mode, skip_chain_length, normalization,  ...
    window_size, lambda_lasso, dict_size, use_pca, nb_components, ...
    random_seed,feat_type);
results_dir = fullfile(output_dir, granularity, results_dir_name, ...
    split_str);
if exist(results_dir, 'dir')
    fprintf('Results already computed in dir: %s\n', results_dir);
    %% If it is final_split, then compute and log average over splits
    if split == final_split
        experiment_dir = fullfile(output_dir, granularity, results_dir_name);
        avg_output_csv_file = [output_csv_file(1:end-4), '_avg.csv'];
        log_average_over_splits(experiment_dir, avg_output_csv_file, granularity);
    end
    return;
end

%% Fit PCA and project data to low dimensional subspace
if use_pca
    nb_training_sequences = length(training_sequences.data);
    nb_testing_sequences = length(testing_sequences.data);

    format_str = ['n_%d_rs_%d'];
    pca_dir_name = sprintf(format_str, nb_components, random_seed);
    pca_dir = fullfile(output_dir, 'pca', granularity, pca_dir_name, split_str);
    pca_matfile = fullfile(pca_dir, 'pca_u.mat');
    if ~exist(pca_dir, 'dir')
        mkdir(pca_dir)
    end

    if ~exist(pca_matfile, 'file')
        fprintf('Saving pca to dir: %s\n', pca_dir);
        fprintf('Fitting PCA...\n');
        % Concatenate training sequences
        X_train_concat = [];

        for i=1:nb_training_sequences
            X_train_concat = [X_train_concat, training_sequences.data{i}];
        end

        [Ud, S] = fit_pca_transform(X_train_concat, nb_components);
        save(pca_matfile, 'Ud', 'S');
    else
        load(pca_matfile);
    end

    fprintf('Projecting data...\n');
    whiten = 0;
    epsilon = 0;
    for cnt = 1 : nb_training_sequences
        x_sequence = training_sequences.data{cnt};
        training_sequences.data{cnt} = pca_transform(x_sequence, ...
            Ud, S, whiten, epsilon);
    end

    for cnt = 1 : nb_testing_sequences
        x_sequence = testing_sequences.data{cnt};
        testing_sequences.data{cnt} = pca_transform(x_sequence, ...
            Ud, S, whiten, epsilon);
    end
end

%% Initialize dictionary via unsupervised learning (lasso)
% Concatenate training data
X_train_concat = [];
nb_training_sequences = length(training_sequences.data);
for i=1:nb_training_sequences
    X_train_concat = [X_train_concat, training_sequences.data{i}];
end
fprintf('Initializing dictionary via unsupervised dictionary learning\n');
psi_init = mexTrainDL(X_train_concat, mex_lasso_param);
psi = psi_init;

%% Compute sparse codes and features
nb_training_sequences = length(training_sequences.data);

training_sequences.sparse_codes = cell(1, nb_training_sequences);
training_sequences.features = cell(1, nb_training_sequences);

for cnt = 1 : nb_training_sequences
    Dhat = training_sequences.data{cnt};
    Uhat = mexLasso(Dhat, psi, mex_lasso_param);
    if strcmp(feat_type, 'default')
        feat = histogram_sparse_codes(Uhat, window_size);
    elseif strcmp(feat_type, 'nodup')
        feat = histogram_sparse_codes_nodup(Uhat, window_size);
    end
    training_sequences.sparse_codes{cnt} = Uhat;
    training_sequences.features{cnt} = feat;
end

nb_testing_sequences = length(testing_sequences.data);
testing_sequences.sparse_codes = cell(1, nb_testing_sequences);
testing_sequences.features = cell(1, nb_testing_sequences);

for cnt = 1 : nb_testing_sequences
    Dhat = testing_sequences.data{cnt};
    Uhat = mexLasso(Dhat, psi, mex_lasso_param);
    feat = histogram_sparse_codes(Uhat, window_size);
    testing_sequences.sparse_codes{cnt} = Uhat;
    testing_sequences.features{cnt} = feat;
end


%% Train CRF
training_labels = training_sequences.labels;
validation_labels = testing_sequences.labels;

%% Normalize data
if strcmp(normalization, 'none')
    training_data = training_sequences.features;
    validation_data = testing_sequences.features;
elseif strcmp(normalization, 'l2pertrial')
    training_normalized_features =  cell(1, nb_training_sequences);
    sequence_cnt = 1;
    for sequence = training_sequences.features
        sequence = sequence{1};
        norms_per_timestep = sqrt(sum(sequence.^2, 1));
        normalized_sequence = sequence / mean(norms_per_timestep);
        training_normalized_features{sequence_cnt} = normalized_sequence;
        sequence_cnt = sequence_cnt + 1;
    end
    testing_normalized_features =  cell(1, nb_testing_sequences);
    sequence_cnt = 1;
    for sequence = testing_sequences.features
        sequence = sequence{1};
        norms_per_timestep = sqrt(sum(sequence.^2, 1));
        normalized_sequence = sequence / mean(norms_per_timestep);
        testing_normalized_features{sequence_cnt} = normalized_sequence;
        sequence_cnt = sequence_cnt + 1;
    end
    training_data = training_normalized_features;
    validation_data = testing_normalized_features;
elseif strcmp(normalization, 'l2')
    avg_l2_norms = zeros(1, nb_training_sequences);
    sequence_cnt = 1;
    for sequence = training_sequences.features
        sequence = sequence{1};
        norms_per_timestep = sqrt(sum(sequence.^2, 1));
        avg_l2_norms(sequence_cnt) = mean(norms_per_timestep);
        sequence_cnt = sequence_cnt + 1;
    end
    avg_l2_norm = mean(avg_l2_norms);
    training_normalized_features =  cell(1, nb_training_sequences);
    sequence_cnt = 1;
    for sequence = training_sequences.features
        sequence = sequence{1};
        normalized_sequence = sequence / avg_l2_norm;
        training_normalized_features{sequence_cnt} = normalized_sequence;
        sequence_cnt = sequence_cnt + 1;
    end
    testing_normalized_features =  cell(1, nb_testing_sequences);
    sequence_cnt = 1;
    for sequence = testing_sequences.features
        sequence = sequence{1};
        normalized_sequence = sequence / avg_l2_norm;
        testing_normalized_features{sequence_cnt} = normalized_sequence;
        sequence_cnt = sequence_cnt + 1;
    end
    training_data = training_normalized_features;
    validation_data = testing_normalized_features;
elseif strcmp(normalization, 'mean_std')
    training_normalized_features =  cell(1, nb_training_sequences);
    sequence_cnt = 1;
    for sequence = training_sequences.features
        sequence = sequence{1};
        Y_temp_mean = mean(sequence,2);
        Y_temp_std = std(sequence');

        % subtract the mean
        sequence_tmp = sequence - Y_temp_mean*ones(1,size(sequence,2));

        % divide each row of the data by it's standard deviation
        for k=1:size(sequence,2)
            sequence_tmp(:,k) = sequence_tmp(:,k)./Y_temp_std';
        end
        training_normalized_features{sequence_cnt} = sequence_tmp;
        sequence_cnt = sequence_cnt + 1;
    end
    testing_normalized_features =  cell(1, nb_testing_sequences);
    sequence_cnt = 1;
    for sequence = testing_sequences.features
        sequence = sequence{1};
        Y_temp_mean = mean(sequence,2);
        Y_temp_std = std(sequence');

        % subtract the mean
        sequence_tmp = sequence - Y_temp_mean*ones(1,size(sequence,2));

        % divide each row of the data by its standard deviation
        for k=1:size(sequence,2)
            sequence_tmp(:,k) = sequence_tmp(:,k)./Y_temp_std';
        end
        testing_normalized_features{sequence_cnt} = sequence_tmp;
        sequence_cnt = sequence_cnt + 1;
    end
    training_data = training_normalized_features;
    validation_data = testing_normalized_features;
end


fprintf('Train CRF\n');
w_init = [];
if strcmp(pairwise_mode, 'pre')
   precomputed_pairwise = precompute_pairwise_from_train(training_labels, ...
       nb_classes, skip_chain_length);
else
   precomputed_pairwise = [];
end

[w, optimization_log] = train_crf(training_data, ...
    training_labels, nb_epochs, reg_c, learning_rate, ...
    batch_size, decay_period, decay_factor, decay_t0, ...
    decay_exponent, use_momentum, momentum, optimizer, ...
    pairwise_mode, precomputed_pairwise, w_init, ...
    validation_data, validation_labels, skip_chain_length, nb_classes, ...
    random_seed);

fprintf('Saving results to dir: %s\n', results_dir);

if ~exist(results_dir, 'dir')
    mkdir(results_dir)
end
save(fullfile(results_dir, 'params.mat'), 'params');

%% Plots
f = figure('visible','off');
plot(optimization_log.training_accuracy);
xlabel('SGD Epochs');
title(sprintf('Training accuracy'));
print(fullfile(results_dir, 'training_accuracy'),'-dpng');

f = figure('visible','off');
plot(optimization_log.training_unreg_loss);
xlabel('SGD Epochs');
title(sprintf('Training unreg loss'));
print(fullfile(results_dir, 'training_unreg_loss'),'-dpng');

f = figure('visible','off');
plot(optimization_log.training_loss);
xlabel('SGD Epochs');
title(sprintf('Training loss'));
print(fullfile(results_dir, 'training_loss'),'-dpng');

f = figure('visible','off');
plot(optimization_log.gradw_norm);
xlabel('SGD Epochs');
ylabel('l2 norm');
title(sprintf('Norm of gradient w.r.t w'));
print(fullfile(results_dir, 'gradw_norm'),'-dpng');

f = figure('visible','off');
plot(optimization_log.w_diff_norm);
xlabel('SGD Epochs');
ylabel('Frobenius norm');
title(sprintf('W_diff norm'));
print(fullfile(results_dir, 'w_diff_norm'),'-dpng');

f = figure('visible','off');
plot(optimization_log.validation_loss);
xlabel('SGD Epochs');
title(sprintf('Validation loss'));
print(fullfile(results_dir, 'validation_loss'),'-dpng');

f = figure('visible','off');
plot(optimization_log.validation_unreg_loss);
xlabel('SGD Epochs');
title(sprintf('Validation unreg loss'));
print(fullfile(results_dir, 'validation_unreg_loss'),'-dpng');

f = figure('visible','off');
plot(optimization_log.validation_accuracy);
xlabel('SGD Epochs');
title(sprintf('Validation accuracy'));
print(fullfile(results_dir, 'validation_accuracy'),'-dpng');

f = figure('visible','off');
plot(optimization_log.training_accuracy, 'b');
hold on;
plot(optimization_log.validation_accuracy, 'r');
hold off;
xlabel('SGD Epochs');
title(sprintf('Accuracy'));
legend('training_acc', 'testing_acc', 'Location', 'best');
print(fullfile(results_dir, 'training_testing_accuracy'),'-dpng');

%% Save parameters, trained weights and optimization logs
save(fullfile(results_dir, 'crf_train_log.mat'), 'w', 'optimization_log');

% %% Save figures to common folder
% tmp_fig_dir = fullfile(output_dir, 'optimization_plots');
% if ~exist(tmp_fig_dir, 'dir')
%     mkdir(tmp_fig_dir)
% end
% 
% fig_prefix = sprintf('%s_', split_str);
% format_str = ['sparse_z_e_%d_c_%.3f_lr_%.4f', ...
%     '_bs_%d_dp_%d_df_%.2f_dt0_%.2f', ...
%     '_dexp_%.2f_um_%d_mom_%.1f_opt_%s_pw_%s', ...
%     '_skip_%d_norm_%s', ...
%     '_win_%d_lambda_%.2f_dict_%d_pca_%d_n_%d', ...
%     '_rs_%d_feat_%s'];
% fig_name = sprintf(format_str, nb_epochs, reg_c, learning_rate, ...
%     batch_size, decay_period, decay_factor, decay_t0, ...
%     decay_exponent, use_momentum, momentum, optimizer, ...
%     pairwise_mode, skip_chain_length, ...
%     window_size, lambda_lasso, dict_size, use_pca, nb_components, ...
%     random_seed, feat_type);
% copyfile(fullfile(results_dir, 'training_loss.png'), ...
%     fullfile(tmp_fig_dir, ...
%     [fig_prefix, fig_name, '_training_loss.png']));
% copyfile(fullfile(results_dir, 'training_accuracy.png'), ...
%     fullfile(tmp_fig_dir, ...
%     [fig_prefix, fig_name, '_training_accuracy.png']));
% copyfile(fullfile(results_dir, 'validation_loss.png'), ...
%     fullfile(tmp_fig_dir, ...
%     [fig_prefix, fig_name, '_validation_loss.png']));
% copyfile(fullfile(results_dir, 'validation_accuracy.png'), ...
%     fullfile(tmp_fig_dir, ...
%     [fig_prefix, fig_name, '_validation_accuracy.png']));
% 
% %% Save figures to common folder
% tmp_fig_dir = fullfile(output_dir, 'optimization_plots');
% if ~exist(tmp_fig_dir, 'dir')
%     mkdir(tmp_fig_dir)
% end

%% Log results and parameters
val_acc = optimization_log.validation_accuracy(end);
val_loss = optimization_log.validation_loss(end);
val_unreg_loss = optimization_log.validation_unreg_loss(end);
train_acc = optimization_log.training_accuracy(end);
train_loss = optimization_log.training_loss(end);
train_unreg_loss = optimization_log.training_unreg_loss(end);

header_str = ['results_dir,granularity,split,val_acc,val_loss,', ...
              'val_unreg_loss,train_acc,train_loss,', ...
              'train_unreg_loss,nb_epochs,reg_c,learning_rate,', ...
              'batch_size,optimizer,decay_period,decay_factor,', ...
              'decay_t0,decay_exponent,use_momentum,momentum,', ...
              'pairwise_mode,skip_chain_length,normalization,random_seed,' ...
              'window_size, lambda_lasso, dict_size, use_pca,', ...
              'nb_components,feat_type'];
if ~exist(output_csv_file, 'file')
    write_header = 1;
else
    write_header = 0;
end

fileID = fopen(output_csv_file, 'a');

if write_header
    % Write header
    fprintf('Writing header\n');
    fprintf(fileID, '%s\n', header_str);
end

row_str_format = ['%s,%s,%d,%f,%f', ...
                  '%f,%f,%f,', ...
                  '%f,%d,%f,%f,', ...
                  '%d,%s,%d,%f,', ...
                  '%d,%d,%d,%f,', ...
                  '%s,%d,%s,%d,', ...
                  '%d,%f,%d,%d,', ...
                  '%d,%s'];
row_str = sprintf(row_str_format, results_dir, granularity, split, val_acc, ...
                  val_loss, val_unreg_loss, train_acc, train_loss, ...
                  train_unreg_loss, nb_epochs, reg_c, learning_rate, ...
                  batch_size, optimizer, decay_period, decay_factor, ...
                  decay_t0, decay_exponent, use_momentum, momentum, ...
                  pairwise_mode, skip_chain_length, normalization, ...
                  random_seed, window_size, lambda_lasso, dict_size, ...
                  use_pca, nb_components, feat_type);
fprintf(fileID, '%s\n', row_str);
fclose(fileID);

%% Also write parameters and results to output folder
fileID = fopen(fullfile(results_dir, 'log.csv'), 'w');
fprintf(fileID, '%s\n', header_str);
fprintf(fileID, '%s\n', row_str);
fclose(fileID);

%% If it is final_split, then compute and log average over splits
if split == final_split
    experiment_dir = fullfile(output_dir, granularity, results_dir_name);
    avg_output_csv_file = [output_csv_file(1:end-4), '_avg.csv'];
    log_average_over_splits(experiment_dir, avg_output_csv_file, granularity);
end

