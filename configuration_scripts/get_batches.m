function [ batches ] = get_batches(list_len, nb_batches)
% Return a cell of nb_batches elements, each one of which contains the 
% indices belonging to the i-th batch of a list with length list_len 

q = floor(list_len / nb_batches); % Only works for positive arguments
r = mod(list_len, nb_batches);

cnt = 0;
indices = zeros(1, nb_batches);
for i = 1 : nb_batches + 1
    indices(i) = q*(i-1) + min(i-1, r);
end

batches = cell(1, nb_batches);
for i = 1 : nb_batches
    batches{i} = [indices(i) + 1 : indices(i+1)];
end

