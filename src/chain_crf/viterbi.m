function [path, max_score, dp_table, backp] = viterbi(unary_potentials, pairwise_potentials)
%% VITERBI Computes optimal sequence of labels using dynamic programming.
% Complexity: timesteps x nb_classes^2
%
% Inputs
% unary_potentials: matrix of size nb_classes x nb_timesteps
% pairwise_potentials: matrix of size nb_classes x nb_classes
% 
% Outputs:
% path: vector of size nb_timesteps with optimal sequence of class labels
% max_score: score of optimal sequence of labels

[nb_classes, nb_timesteps] = size(unary_potentials);

dp_table = unary_potentials;
backp = zeros(nb_classes, nb_timesteps);
path = zeros(1, nb_timesteps);

%% Forward pass
for t = 2 : nb_timesteps
    for k = 1 : nb_classes
        % Find the score of the optimal path ending in class k at timestep
        % t
        % score(k, t) = max_j score(j, t-1) + unary_potentials(k,t) +
        % pairwise_potentials(j, k). 
        % Since unary_potentials(k,t) is a common term over all j, this
        % is equivalent to score(k, t) = max_j score(j, t-1) +
        % pairwise_potentials(j, k). 
        [max_val, max_ind] = max(dp_table(:,t-1) + ...
            pairwise_potentials(:,k));
        dp_table(k, t) = max_val + unary_potentials(k, t);
        backp(k, t) = max_ind;
    end
end

%% Backward pass
%path(nb_timesteps) = max(backp(:, nb_timesteps));
[max_score, max_ind] = max(dp_table(:, nb_timesteps));
path(nb_timesteps) = max_ind;
%max_score = max(dp_table(:, nb_timesteps));
for t = nb_timesteps-1:-1:1
    path(t) = backp(path(t+1), t+1);
end

end
