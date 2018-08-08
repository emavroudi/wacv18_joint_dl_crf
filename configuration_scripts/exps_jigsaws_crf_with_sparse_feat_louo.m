function param_grid_jigsaws_crf_with_sparse_feat_louo(batch_id, nb_batches, ...
    dataset_name, model_name, output_dir, output_csv_file, code_path, ...
    spams_matlab_path, datasets_dir)

addpath(fullfile(spams_matlab_path, 'spams-matlab', 'test_release'));
addpath(fullfile(spams_matlab_path, 'spams-matlab', 'src_release'));
addpath(fullfile(spams_matlab_path, 'spams-matlab', 'build'));
addpath(genpath(code_path));

%% Parameters to be specified by user
dataset_name_lst = {dataset_name}; % JIGSAWS, FiftySalads
model_name_lst = {model_name}; % crf_with_raw_feat, crf_with_sparse_feat, joint_dl_crf
task_lst = {'Needle_Passing'};
setup_lst = {'LOUO'};
granularity_lst = {'eval'};
standardize_lst = {1};
feat_type_lst = {'default'};
input_type_lst = {'default'};
nb_epochs_lst = {100};
reg_c_lst = {1};
learning_rate_lst = {0.0001 0.001};
batch_size_lst = {1};
decay_period_lst = {20};
decay_factor_lst = {0.5};
decay_t0_lst = {10};
decay_exponent_lst = {1};
use_momentum_lst = {1};
momentum_lst = {0.9};
optimizer_lst = {'block_decay'};
pairwise_mode_lst = {'joint'};
skip_chain_length_lst = {1};
window_size_lst = {51 71 81};
lambda_lasso_lst = {0.1 0.3 0.5};
dict_size_lst = {100 150 200};
use_pca_lst = {1};
nb_components_lst = {35};
gradpsi_scaling_factor = {1};
split_lst = {3, 4, 5, 6, 7};
random_seed_lst = {42};

%% Generate parameter grid: cell nb_experiments x nb_variables,
%  variables are ordered as the arguments of allcomb
addpath('./allcomb')
A = allcomb(dataset_name_lst, model_name_lst, task_lst, setup_lst, ...
    granularity_lst, standardize_lst, feat_type_lst, input_type_lst, nb_epochs_lst, ...
    reg_c_lst, learning_rate_lst, batch_size_lst, decay_period_lst, ...
    decay_factor_lst, decay_t0_lst, decay_exponent_lst, use_momentum_lst, ...
    momentum_lst, optimizer_lst, pairwise_mode_lst, skip_chain_length_lst, ...
    window_size_lst, lambda_lasso_lst, dict_size_lst, use_pca_lst, nb_components_lst, ...
    gradpsi_scaling_factor, random_seed_lst, split_lst);


nb_experiments = size(A, 1)
fprintf('Nb experiments: %d\n', nb_experiments);
batches = get_batches(nb_experiments, nb_batches);
experiment_ids = batches{batch_id};

for experiment_id = experiment_ids
    params.dataset_name = A{experiment_id, 1};
    params.model_name = A{experiment_id, 2};
    params.task = A{experiment_id, 3};
    params.setup = A{experiment_id, 4};
    params.granularity = A{experiment_id, 5};
    params.standardize = A{experiment_id, 6};
    params.feat_type = A{experiment_id, 7};
    params.input_type = A{experiment_id, 8};
    params.nb_epochs = A{experiment_id, 9};
    params.reg_c = A{experiment_id, 10};
    params.learning_rate = A{experiment_id, 11};
    params.batch_size = A{experiment_id, 12};
    params.decay_period = A{experiment_id, 13};
    params.decay_factor = A{experiment_id, 14};
    params.decay_t0 = A{experiment_id, 15};
    params.decay_exponent = A{experiment_id, 16};
    params.use_momentum = A{experiment_id, 17};
    params.momentum = A{experiment_id, 18};
    params.optimizer = A{experiment_id, 19};
    params.pairwise_mode = A{experiment_id, 20};
    params.skip_chain_length = A{experiment_id, 21};
    params.window_size = A{experiment_id, 22};
    params.lambda_lasso = A{experiment_id, 23};
    params.dict_size = A{experiment_id, 24};
    params.use_pca = A{experiment_id, 25};
    params.nb_components = A{experiment_id, 26};
    params.gradpsi_scaling_factor = A{experiment_id, 27};
    params.random_seed = A{experiment_id, 28};
    params.split = A{experiment_id, 29};
    params.datasets_dir = datasets_dir;
    experiments_summer17_main(output_dir, output_csv_file, params);    
end
