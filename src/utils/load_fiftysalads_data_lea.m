function [training_sequences, testing_sequences, split_str, classes] = load_fiftysalads_data_lea(...
    data_dir, granularity, split, standardize)

if ~(strcmp(granularity, 'eval') || strcmp(granularity, 'mid') || ...
        strcmp(granularity, 'eval_cross_val') || ...
        strcmp(granularity, 'mid_cross_val'))
    error('Not supported granularity');
end

split_str = sprintf('%s-%d', granularity, split);

%% Get training/testing sequences data filenames
fid = fopen(fullfile(data_dir, 'lea_splits', ...
    ['Split_' num2str(split)], 'train.txt'));
C = textscan(fid, '%s');
training_video_ids = C{1};
fclose(fid);

fid = fopen(fullfile(data_dir, 'lea_splits', ...
    ['Split_' num2str(split)], 'test.txt'));
C = textscan(fid, '%s');
testing_video_ids = C{1};
fclose(fid);

nb_training_samples = length(training_video_ids);
nb_testing_samples = length(testing_video_ids);

% Features are the same for all split (just skeleton data),
% just replicated in splits feat folders
if strcmp(granularity, 'eval') || strcmp(granularity, 'mid')
    feat_dir = fullfile(data_dir, 'SpatialCNN_feat', ...
        ['SpatialCNN_' granularity], ['Split_' num2str(split)]);
elseif strcmp(granularity, 'eval_cross_val') 
    feat_dir = fullfile(data_dir, 'SpatialCNN_feat', ...
        ['SpatialCNN_eval'], ['Split_' num2str(1)]);
elseif strcmp(granularity, 'mid_cross_val')
    feat_dir = fullfile(data_dir, 'SpatialCNN_feat', ...
        ['SpatialCNN_mid'], ['Split_' num2str(1)]);
end
%% Build list of training/testing sample sequences
unique_labels = [];
for cnt = 1 : nb_training_samples
    input_matfilename = sprintf('rgb-%s.avi.mat', training_video_ids{cnt});
    input_matfile = fullfile(feat_dir, input_matfilename);
    load(input_matfile);
    % x_sequence: feat_dim (30) x nb_timesteps
    x_sequence = S;
    % y_sequence: 1 x nb_timesteps
    y_sequence = double(Y' + 1);
    
    if standardize == 1
        x_sequence = zscore(x_sequence,0,2);
    end
    training_sequences.data{cnt} = x_sequence;
    training_sequences.labels{cnt} = y_sequence;
    unique_labels = unique([unique_labels y_sequence]);
end

for cnt = 1 : nb_testing_samples
    input_matfilename = sprintf('rgb-%s.avi.mat', testing_video_ids{cnt});
    input_matfile = fullfile(feat_dir, input_matfilename);
    load(input_matfile);
    % x_sequence: feat_dim (30) x nb_timesteps
    x_sequence = S;
    % y_sequence: 1 x nb_timesteps
    y_sequence = double(Y' + 1);
    
    if standardize == 1
        x_sequence = zscore(x_sequence,0,2);
    end
    testing_sequences.data{cnt} = x_sequence;
    testing_sequences.labels{cnt} = y_sequence;
    unique_labels = unique([unique_labels y_sequence]);
end

classes = unique_labels;

%% If crossval redistribute videos to training/testing (needs refactoring)
if strcmp(granularity, 'eval_cross_val') || ...
        strcmp(granularity, 'mid_cross_val')
    
    % Fixed parameters for now
    cut_segments = 1;
    segments_duration = 0.8;
    k_fold = 5;
    
    if split > k_fold
        error('Invalid split %d', split);
    end
    
    % Collect all sequences
    cnt = 1;
    for i = 1 : length(training_sequences.data)
        all_sequences.data{cnt} = training_sequences.data{i};
        all_sequences.labels{cnt} = training_sequences.labels{i};
        cnt = cnt + 1;
    end
    for i = 1 : length(testing_sequences.data)
        all_sequences.data{cnt} = testing_sequences.data{i};
        all_sequences.labels{cnt} = testing_sequences.labels{i};
        cnt = cnt + 1;
    end
    
    % Get cross-validation partition in training/testing
    nb_sequences = length(all_sequences.data);
    sequence_indices = 1:nb_sequences;
    indices = crossvalind('Kfold', sequence_indices, k_fold);
    test_logic_ind = indices == split;
    train_logic_ind = ~test_logic_ind;
    training_indices = sequence_indices(train_logic_ind);
    testing_indices = sequence_indices(test_logic_ind);
    
    % Create training_sequences, testing_sequences
    training_sequences.data = {};
    training_sequences.labels = {};
    testing_sequences.data = {};
    testing_sequences.labels = {};
    if ~cut_segments
        cnt = 1;
        for train_ind = training_indices
            training_sequences.data{cnt} = all_sequences.data{train_ind};
            training_sequences.labels{cnt} = all_sequences.labels{train_ind};
            cnt = cnt + 1;
        end
        cnt = 1;
        for test_ind = testing_indices
            testing_sequences.data{cnt} = all_sequences.data{test_ind};
            testing_sequences.labels{cnt} = all_sequences.labels{test_ind};
            cnt = cnt + 1;
        end
    else
        cnt = 1;
        for train_ind = training_indices
            nb_timesteps = size(all_sequences.data{train_ind},2);
            segment_length = floor(segments_duration*nb_timesteps);
            start_frame = randi(nb_timesteps-segment_length);
            end_frame = start_frame + segment_length - 1;
            training_sequences.data{cnt} = ...
                all_sequences.data{train_ind}(:, start_frame:end_frame);
            training_sequences.labels{cnt} = ...
                all_sequences.labels{train_ind}(start_frame:end_frame);
            cnt = cnt + 1;
        end
        cnt = 1;
        for test_ind = testing_indices
            nb_timesteps = size(all_sequences.data{test_ind},2);
            segment_length = floor(segments_duration*nb_timesteps);
            start_frame = randi(nb_timesteps-segment_length);
            end_frame = start_frame + segment_length - 1;
            testing_sequences.data{cnt} = ...
                all_sequences.data{test_ind}(:, start_frame:end_frame);
            testing_sequences.labels{cnt} = ...
                all_sequences.labels{test_ind}(start_frame:end_frame);
            cnt = cnt + 1;
        end
        
    end
    
end

end
