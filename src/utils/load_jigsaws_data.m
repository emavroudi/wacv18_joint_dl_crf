function [training_sequences, testing_sequences, split_str, classes] = load_jigsaws_data(...
    data_dir, task, setup, split)

rootfolder = fullfile(data_dir, task);
trials = [1, 2, 3, 4, 5];
subjects= {'B',  'C',    'D',    'E',    'F',    'G',    'H',    'I'};

if strcmp(setup, 'LOUO')
    split_str = sprintf('%s-%s', setup, subjects{split});
elseif strcmp(setup, 'LOSO')
    split_str = sprintf('%s-%d', setup, trials(split));
elseif strcmp(setup, 'cross_val')
    split_str = sprintf('%s-%d', setup, split);
else
    error('Not supported setup %s', setup);
end

%% Determine training/testing subjects/trials
if strcmp(setup,'LOUO')       
    
    subject_indices = 1:8;
        
    if strcmp(task,'Needle_Passing') && split > 5
        % Needle_Passing does not have subject G (5)
        subj_test_idx = split + 1;
    else
        subj_test_idx = split;
    end
    
    subj_train = subjects(subject_indices(subject_indices~=subj_test_idx));
    subj_test = subjects(subj_test_idx);
    trial_train = trials;
    trial_test = trials;
elseif strcmp(setup, 'LOSO')
    
    trial_train = trials(trials~=split);
    trial_test = split;
    subj_train = subjects;
    subj_test = subjects;
elseif strcmp(setup, 'cross_val')
    % This do not affect the final cross validation which will
    % generate new splits, irrespective of trials/subjects
    % It just has to be the same for all splits
    trial_train = trials(1:4);
    trial_test = 5;
    subj_train = subjects;
    subj_test = subjects;    
else
    error('Invalid setup');
end
%% Load training/testing data
fprintf('Reading training data..\n');
[X_train,G_train] = read_data(task,subj_train,trial_train,rootfolder);
% standardize data trial by trial
X_train = standardize_data(X_train);

fprintf('Reading testing data..\n');
[X_test,G_test] = read_data(task,subj_test,trial_test,rootfolder);
% standardize data trial by trial
X_test = standardize_data(X_test);

[nb_training_subjects, nb_training_trials] = size(X_train);
[nb_testing_subjects, nb_testing_trials] = size(X_test);

%% Get classes ids (possibly non contiguous)
labels_concat = [];
for subj = 1 : nb_training_subjects
    for trial = 1 : nb_training_trials
        if isempty(X_train{subj, trial})
            continue;
        end
        y_sequence = G_train{subj, trial};
        labels_concat = [labels_concat y_sequence];
    end
end

for subj = 1 : nb_testing_subjects
    for trial = 1 : nb_testing_trials
        if isempty(X_test{subj, trial})
            continue;
        end
        y_sequence = G_test{subj, trial};
        labels_concat = [labels_concat y_sequence];
    end
end

classes = unique(labels_concat);

%% Build list of training/testing sample sequences

cnt = 1;
for subj = 1 : nb_training_subjects
    for trial = 1 : nb_training_trials
        if isempty(X_train{subj, trial})
            continue;
        end
        x_sequence = X_train{subj, trial};
        y_sequence = G_train{subj, trial};
        y_sequence = make_labels_contiguous(y_sequence, classes);
        
        training_sequences.data{cnt} = x_sequence;
        training_sequences.labels{cnt} = y_sequence;
        cnt = cnt + 1;
    end
end

cnt = 1;
for subj = 1 : nb_testing_subjects
    for trial = 1 : nb_testing_trials
        if isempty(X_test{subj, trial})
            continue;
        end
        
        x_sequence = X_test{subj, trial};
        y_sequence = G_test{subj, trial};
        y_sequence = make_labels_contiguous(y_sequence, classes);
        
        testing_sequences.data{cnt} = x_sequence;
        testing_sequences.labels{cnt} = y_sequence;
        cnt = cnt + 1;
    end
end

%% If crossval redistribute videos to training/testing (needs refactoring)
if strcmp(setup, 'cross_val')
    
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
