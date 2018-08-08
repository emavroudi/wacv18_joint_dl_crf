function Y_normalized = standardize_data(Y)

[m,n] = size(Y);

for i=1:m
    for j=1:n
        if ~isempty(Y{i,j})
            clear Y_temp
            Y_temp = Y{i,j};
            Y_temp_mean = mean(Y_temp,2);
            Y_temp_std = std(Y_temp');
            
            % subtract the mean
            Y{i,j} = Y{i,j} - Y_temp_mean*ones(1,size(Y{i,j},2));

            % divide each row of the data by it's standard deviation
            for k=1:size(Y{i,j},2)
                Y{i,j}(:,k) = Y{i,j}(:,k)./Y_temp_std';
            end
        end
    end
Y_normalized = Y;
end