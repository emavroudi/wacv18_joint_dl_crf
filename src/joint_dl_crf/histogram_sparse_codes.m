function [z] = histogram_sparse_codes(Uhat,window_size)
%% HISTOGRAM_SPARSE_CODES
% Compute histogram of positive and negative sparse codes using 
% a sliding window with stride 1 and length window_size. 

% temporal window for sum-pooling
template = ones(1,window_size);

%% sum-pooling of sparse codes (after duplicating the sparse codes to positive and negative components)

Uhat_pos = +full(Uhat>0);
Uhat_neg = +full(Uhat<0);

% duplicating the sparse codes to positive and negative components
Uhat_pos = full(Uhat_pos.*Uhat);

Uhat_neg = full(Uhat_neg.*Uhat); % negative components keep their sign (FEATURE III.B)

hist_columns_pos = [];
hist_columns_neg = [];

hist_columns_pos = conv2(Uhat_pos, template, 'same');
hist_columns_neg = conv2(Uhat_neg, template, 'same');


hist_columns_signs = [hist_columns_pos;hist_columns_neg];

z = hist_columns_signs;

z = z/window_size;

