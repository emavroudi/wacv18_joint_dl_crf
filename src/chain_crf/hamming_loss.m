function loss = hamming_loss(y_sequence_true, y_sequence_pred)
%% HAMMING_LOSS
% 
% Inputs
% y_sequence_true: ground truth sequence of labels
% y_sequence_pred: predicted sequence of labels
%
% Output
% hamming loss: sum of wrong predicted labels

loss = sum(y_sequence_pred ~= y_sequence_true);

end