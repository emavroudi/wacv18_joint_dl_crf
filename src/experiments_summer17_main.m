function experiments_summer17_main(output_dir, output_csv_file, params)

if strcmp(params.dataset_name, 'JIGSAWS')
    params.data_dir = fullfile(params.datasets_dir, 'JIGSAWS');

    if strcmp(params.setup, 'LOUO')
        if strcmp(params.task, 'Needle_Passing')
            params.final_split = 7;
        else
            params.final_split = 8;
        end
    elseif strcmp(params.setup, 'LOSO')
        params.final_split = 5;
    elseif strcmp(params.setup, 'cross_val')
        params.final_split = 5;
    else
        error('Not supported setup %s', params.setup);
    end

    if strcmp(params.model_name, 'crf_with_raw_feat')
        jigsaws_crf_with_raw_feat(output_dir, output_csv_file, params);
    elseif strcmp(params.model_name, 'crf_with_sparse_feat')
        jigsaws_crf_with_sparse_feat(output_dir, output_csv_file, params);
    elseif strcmp(params.model_name, 'joint_dl_crf')
        jigsaws_joint_dl_crf(output_dir, output_csv_file, params);
    else 
        error('Not supported model: %s', params.model_name);
    end

elseif strcmp(params.dataset_name, 'FiftySalads')
    params.data_dir = fullfile(params.datasets_dir, '50Salads');
    params.final_split = 5;

    if strcmp(params.model_name, 'crf_with_raw_feat')
        fiftysalads_crf_with_raw_feat(output_dir, output_csv_file, params);
    elseif strcmp(params.model_name, 'crf_with_sparse_feat')
        fiftysalads_crf_with_sparse_feat(output_dir, output_csv_file, params);
    elseif strcmp(params.model_name, 'joint_dl_crf')
        fiftysalads_joint_dl_crf(output_dir, output_csv_file, params);
    else 
        error('Not supported model: %s', params.model_name)
    end

else
    error('Not supported dataset_name %s', params.dataset_name);
end

end

