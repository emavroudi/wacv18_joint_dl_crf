function pairwise_mat = precompute_pairwise(G, classes, skip_chain)


[nb_subjects,nb_trials] = size(G);
numStates = numel(classes);
tr = zeros(numStates);

for i=1:nb_subjects
    for j=1:nb_trials
        for k=1:skip_chain
            if ~isempty(G{i,j})
                states=G{i,j}(k:skip_chain:end);
                no_classes = numel(classes);
                classes_idx = 1:no_classes;
                for ii=1:no_classes
                    states(states==classes(ii))=classes_idx(ii);
                end
                % count up the transitions from the state path
                for count=1:numel(states)-1
                    tr(states(count),states(count+1)) = ...
                        tr(states(count),states(count+1)) + 1;
                end

            end
            clear states
        end
    end
end

trRowSum = sum(tr,2);

% if we don't have any values then report zeros instead of NaNs.
trRowSum(trRowSum == 0) = -inf;

% normalize to give frequency estimate.
tr = tr./repmat(trRowSum,1,numStates);
pairwise_mat = log(tr);
pairwise_mat(isinf(pairwise_mat))=-1e6;
