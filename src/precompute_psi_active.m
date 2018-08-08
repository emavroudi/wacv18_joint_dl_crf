function [psi_active_per_frame, A_per_frame, activeset_per_frame, ...
    activeset_mod_per_frame] = precompute_psi_active(Uhat, psi)

nb_timesteps = size(Uhat, 2);
[~, Uhat_posneg] =  sumpool_dup(Uhat);

psi_active_per_frame = cell(1, nb_timesteps);
A_per_frame = cell(1, nb_timesteps);
activeset_per_frame = cell(1, nb_timesteps);
activeset_mod_per_frame = cell(1, nb_timesteps);

parfor t_par = 1 : nb_timesteps
    u = Uhat_posneg(:, t_par);
    
    % u: vector of concatenated positive sparse codes u_+ and negative
    %    sparse codes u_- with size: (2*dict_size, 1)
    % Find indices of non-zero sparse codes
    activeset = find(u~=0);
    % Find indices of non-zero sparse codes in the original sparse
    % coefficients vector. For example, for dict_size=200, non-zero coefficient
    % in u, with index 356 corresponds to non-zero (negative) coefficient
    % of original sparse coefficients vector with index 156.
    activeset_mod = mod(activeset,size(psi,2));
    % Fix index of last sparse coefficient (if non-zero), modulo can
    % give indices 0 - dict_size - 1.
    activeset_mod(activeset_mod==0) = size(psi,2);
    % Get active columns of psi
    psi_active = psi(:,activeset_mod);
    % Compute A
    A = inv(psi_active'*psi_active);
    
    psi_active_per_frame{t_par} = psi_active;
    A_per_frame{t_par} = A;
    activeset_per_frame{t_par} = activeset;
    activeset_mod_per_frame{t_par} = activeset_mod;
end
