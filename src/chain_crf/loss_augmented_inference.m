function [y_sequence_hat] = loss_augmented_inference(unary_potentials, ...
    pairwise_potentials, y_sequence, skip_chain_length)

augmented_unary_potentials = loss_augmented_unaries(unary_potentials, ...
    y_sequence);

if skip_chain_length == 1
    [y_sequence_hat, max_score, dp_table, backp] = viterbi(augmented_unary_potentials, ...
        pairwise_potentials);
else
    nb_timesteps = size(augmented_unary_potentials, 2);
    path = zeros(1, nb_timesteps);
    for k = 1 : skip_chain_length
        [partial_path, max_score, dp_table, backp]  = viterbi(augmented_unary_potentials(:, k:skip_chain_length:end), ...
            pairwise_potentials);
        path(k:skip_chain_length:end) = partial_path;
    end
    y_sequence_hat = path;
end

end
