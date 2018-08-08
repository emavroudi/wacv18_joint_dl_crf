function log_average_over_splits(experiment_dir, output_csv_file, setup)

subdirs = get_subdirs(experiment_dir, setup);
nb_splits = length(subdirs);

accs = zeros(1, nb_splits);
for i = 1 : nb_splits
    results_dir = fullfile(experiment_dir, subdirs{i});
    load(fullfile(results_dir, 'crf_train_log.mat'));
    accs(i) = optimization_log.validation_accuracy(end);
end
avg_acc = mean(accs);
header_str = ['experiment_dir,setup,avg_acc,nb_splits'];

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

fprintf('Average validation accuracy over all splits: %f for experiment: %s\n', ...
        avg_acc, experiment_dir);
row_str_format = ['%s,%s,%f,%d'];
row_str = sprintf(row_str_format, experiment_dir, setup, avg_acc, nb_splits);
fprintf(fileID, '%s\n', row_str);
fclose(fileID);

