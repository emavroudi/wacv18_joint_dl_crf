function [z] = histogram_sparse_codes_nodup(Uhat,window_size)
%% HISTOGRAM_SPARSE_CODES_NODUP
% Compute histogram of sparse codes using 
% a sliding window with stride 1 and length window_size. 

% temporal window for sum-pooling
template = ones(1,window_size);

%% sum-pooling of sparse codes
%% Uncomment for ReLU
%Uhat_pos = +full(Uhat>0);
%Uhat_pos = full(Uhat_pos.*Uhat);
%Uhat = Uhat_pos;

Uhat = +full(Uhat);

hist_columns = conv2(Uhat, template, 'same');

z = hist_columns/window_size;

