function jigsaws_joint_dl_crf(output_dir, output_csv_file, params)

%% Parameters
task = params.task;
setup = params.setup;
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
window_size = params.window_size;
lambda_lasso = params.lambda_lasso;
dict_size = params.dict_size;
use_pca = params.use_pca;
nb_components = params.nb_components;
data_dir = params.data_dir;
input_type = params.input_type; % 'default', 'dipietro'
feat_type = params.feat_type; % 'default', 'nodup'
gradpsi_scaling_factor = params.gradpsi_scaling_factor;
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
if strcmp(input_type, 'default')
    [training_sequences, testing_sequences, split_str, classes] = load_jigsaws_data(...
        data_dir, task, setup, split);
else
    error('Not supported input_type %s', input_type);
end
nb_classes = length(classes);

format_str = ['joint_dl_crf_e_%d_c_%.3f_lr_%.4f', ...
    '_bs_%d_dp_%d_df_%.2f_dt0_%.2f_dexp_%.2f_um_%d_mom_%.1f_opt_%s_pw_%s', ...
    '_skip_%d_win_%d_lambda_%.2f_dict_%d_pca_%d_n_%d_rs_%d_scf_%f_feat_%s_input_%s'];

results_dir_name = sprintf(format_str, nb_epochs, reg_c, learning_rate, ...
    batch_size, decay_period, decay_factor, decay_t0, ...
    decay_exponent, use_momentum, momentum, optimizer, ...
    pairwise_mode, skip_chain_length, window_size, lambda_lasso, dict_size, ...
    use_pca, nb_components, random_seed, ...
    gradpsi_scaling_factor, feat_type, input_type);
results_dir = fullfile(output_dir, task, results_dir_name, ...
    split_str);
if exist(results_dir, 'dir')
    fprintf('Results already computed in dir: %s\n', results_dir);
    %% If it is final_split, then compute and log average over splits
    if split == final_split
        experiment_dir = fullfile(output_dir, task, results_dir_name);
        avg_output_csv_file = [output_csv_file(1:end-4), '_avg.csv'];
        log_average_over_splits(experiment_dir, avg_output_csv_file, setup);
    end
    return;
end

%% Fit PCA and project data to low dimensional subspace
if use_pca
    nb_training_sequences = length(training_sequences.data);
    nb_testing_sequences = length(testing_sequences.data);

    format_str = 'n_%d_rs_%d_input_%s';
    pca_dir_name = sprintf(format_str, nb_components, random_seed, input_type);
    pca_dir = fullfile(output_dir, task, 'pca', pca_dir_name, split_str);
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

%% Jointly train dictionary and CRF
training_labels = training_sequences.labels;
validation_labels = testing_sequences.labels;
training_data = training_sequences.data;
validation_data = testing_sequences.data;

fprintf('Jointly Train Dict and CRF\n');
w_init = [];
if strcmp(pairwise_mode, 'pre')
    precomputed_pairwise = precompute_pairwise_from_train(training_labels, ...
        nb_classes, skip_chain_length);
else
    precomputed_pairwise = [];
end

if strcmp(feat_type, 'default')
    [w, psi, optimization_log] =  train_dict_crf_scaled(...
        training_data, training_labels, ...
        validation_data, validation_labels, nb_epochs, reg_c, learning_rate, ...
        batch_size, decay_period, decay_factor, decay_t0, ...
        decay_exponent, use_momentum, momentum, optimizer, ...
        pairwise_mode, precomputed_pairwise, w_init, ...
        skip_chain_length, nb_classes, ...
        psi_init, window_size, mex_lasso_param, random_seed, ...
        gradpsi_scaling_factor);
elseif strcmp(feat_type, 'nodup')
    [w, psi, optimization_log] =  train_dict_crf_scaled_nodup(...
        training_data, training_labels, ...
        validation_data, validation_labels, nb_epochs, reg_c, learning_rate, ...
        batch_size, decay_period, decay_factor, decay_t0, ...
        decay_exponent, use_momentum, momentum, optimizer, ...
        pairwise_mode, precomputed_pairwise, w_init, ...
        skip_chain_length, nb_classes, ...
        psi_init, window_size, mex_lasso_param, random_seed, ...
        gradpsi_scaling_factor);
else
    error('Not supported feat_type %s', feat_type);
end


if ~exist(results_dir, 'dir')
    mkdir(results_dir)
end

fprintf('Saving results to dir: %s\n', results_dir);
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
save(fullfile(results_dir, 'crf_train_log.mat'), 'w', 'psi', 'optimization_log');

%% Log results and parameters
val_acc = optimization_log.validation_accuracy(end);
val_loss = optimization_log.validation_loss(end);
val_unreg_loss = optimization_log.validation_unreg_loss(end);
train_acc = optimization_log.training_accuracy(end);
train_loss = optimization_log.training_loss(end);
train_unreg_loss = optimization_log.training_unreg_loss(end);

header_str = ['results_dir,task,setup,split,val_acc,val_loss,', ...
    'val_unreg_loss,train_acc,train_loss,', ...
    'train_unreg_loss,nb_epochs,reg_c,learning_rate,', ...
    'batch_size,optimizer,decay_period,decay_factor,', ...
    'decay_t0,decay_exponent,use_momentum,momentum,', ...
    'pairwise_mode,skip_chain_length,random_seed,' ...
    'window_size,lambda_lasso,dict_size,use_pca,', ...
    'nb_components,gradpsi_scaling_factor,feat_type,input_type'];
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

row_str_format = ['%s,%s,%s,%d,%f,%f,', ...
    '%f,%f,%f,', ...
    '%f,%d,%f,%f,', ...
    '%d,%s,%d,%f,', ...
    '%d,%d,%d,%f,', ...
    '%s,%d,%d,', ...
    '%d,%f,%d,%d,', ...
    '%d,%f,%s,%s'];

row_str = sprintf(row_str_format, results_dir, task, setup, split, val_acc, ...
    val_loss, val_unreg_loss, train_acc, train_loss, ...
    train_unreg_loss, nb_epochs, reg_c, learning_rate, ...
    batch_size, optimizer, decay_period, decay_factor, ...
    decay_t0, decay_exponent, use_momentum, momentum, ...
    pairwise_mode, skip_chain_length,  ...
    random_seed, window_size, lambda_lasso, dict_size, ...
    use_pca, nb_components, gradpsi_scaling_factor, feat_type, input_type);
disp(row_str);
fprintf(fileID, '%s\n', row_str);
fclose(fileID);

%% Also write parameters and results to output folder
fileID = fopen(fullfile(results_dir, 'log.csv'), 'w');
fprintf(fileID, '%s\n', header_str);
fprintf(fileID, '%s\n', row_str);
fclose(fileID);

%% If it is final_split, then compute and log average over splits
if split == final_split
    experiment_dir = fullfile(output_dir, task, results_dir_name);
    avg_output_csv_file = [output_csv_file(1:end-4), '_avg.csv'];
    log_average_over_splits(experiment_dir, avg_output_csv_file, setup);
end
