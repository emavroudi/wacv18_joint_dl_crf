function res = loss_augmented_unaries(unary_potentials, y_sequence)
%% LOSS_AUGMENTED_UNARIES 
%
% Inputs
%
% unary_potentials: matrix with size (nb_classes x nb_timesteps)
% y_sequence: ground truth label sequence (vector of size nb_timesteps)
% 
% Output
% res: loss augmented unary potentials (assuming hamming loss)


[nb_classes, nb_timesteps] = size(unary_potentials);
% Add 1 to all classes different from the ground truth class per timestep
% For efficiency we add 1 to all elements of the unary_potentials matrix
% and then subtract 1 from the elements corresponding to the ground 
% truth class per timestep
res = unary_potentials + 1; 
for t = 1 : nb_timesteps
    res(y_sequence(t), t) = res(y_sequence(t), t) - 1; 
end

end
