function [y_sequence_hat] = inference(unary_potentials, ...
    pairwise_potentials, skip_chain_length)
% Inputs
% unary_potentials: matrix of size nb_classes x nb_timesteps
% pairwise_potentials: matrix of size nb_classes x nb_classes
% 

if skip_chain_length == 1
    [y_sequence_hat, max_score, dp_table, backp] = viterbi(unary_potentials, ...
        pairwise_potentials);
else
    nb_timesteps = size(unary_potentials, 2);
    path = zeros(1, nb_timesteps);
    for k = 1 : skip_chain_length
        [partial_path, max_score, dp_table, backp]  = viterbi(unary_potentials(:, k:skip_chain_length:end), ...
            pairwise_potentials);
        path(k:skip_chain_length:end) = partial_path;
        %dp_table
        %backp
    end
    y_sequence_hat = path;
end
end
