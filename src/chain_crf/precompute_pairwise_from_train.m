function pairwise_mat = precompute_pairwise_from_train(training_labels, ...
    nb_classes, skip_chain)
% Assumes contiguous training labels (1...C)

numStates = nb_classes;
tr = zeros(numStates);
nb_training_sequences = length(training_labels);

for i = 1 : nb_training_sequences
    
    sequence_labels = training_labels{i};
    
    for k=1:skip_chain
        
        states=sequence_labels(k:skip_chain:end);
        % count up the transitions from the state path
        for count=1:numel(states)-1
            tr(states(count),states(count+1)) = ...
                tr(states(count),states(count+1)) + 1;
        end
        
        clear states
    end
end

tr
trRowSum = sum(tr,2);

% if we don't have any values then report zeros instead of NaNs.
trRowSum(trRowSum == 0) = -inf;

% normalize to give frequency estimate.
tr = tr./repmat(trRowSum,1,numStates);
pairwise_mat = log(tr);
pairwise_mat(isinf(pairwise_mat))=-1e6;
