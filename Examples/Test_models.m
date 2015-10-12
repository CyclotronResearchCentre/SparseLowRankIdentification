% This is a generic file for calling the ADMM algorithm
% for sparse plus low-rank autoregressive identification
% on data coming from an AR(1) model with a well-known sparsity pattern 
%
% Refer "Sparse plus low-rank autoregressive identification in neuroimaging time series",
% Raphaël Liégeois, Bamdev Mishra, Mattia Zorzi, and Rodolphe Sepulchre,
% Technical report, arXiv:1503.08639, 2015.
%
% This implementation is due to
% Raphaël Liégeois <R.Liegeois@ulg.ac.be> and
% Bamdev Mishra <b.mishra@ulg.ac.be>, 2015.

clc; clear all; close all;


%% Test models and the values of lambda and gamma that give good results

load('Dynamic_model_size_10.mat'); lambda =  0.63; gamma = 0.71;
% load('Dynamic_model_size_100.mat'); lambda = 1.08; gamma = 0.26;






% n       is the number of manifest variables
% p       is the number of lags
% Omega   is the true sparsity pattern
% h       is the number of latent variables
% A       encodes the AR model

%% Generate data from this model
N_tr = 1000;   % Number of first points not considered in final data because corresponding to transitory behavior
N_perm = 1000; % Number of time points


e = randn(n + h, N_tr + N_perm);
X = zeros(n + h, N_tr + N_perm);
for k = 1 : N_tr + N_perm - 1
    X(:,k+1) = A(:,:,1)\(-A(:,:,2)*X(:,k)+e(:,k));
end
X = X(:,N_tr+1:end);
X_m = X(1:n,:);

%% Other parameters

p = 1; %Number of lags

% Computation of C from X_m and for the number of lags that are asked.
temp = xcorr(X_m', p,'biased'); %If 'unbiased', no guarantee that C is symmetric positive definite!

C_line = zeros(n,n,2*p+1);
for i = -p:p;
    C_line(:,:,i+p+1)=reshape(temp(i+p+1,:),n,n);
end
C = zeros(n*(p+1));
for i=1:p+1
    for j=1:p+1
        C(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=C_line(:,:,(j-i)+p+1);
    end
end


%% ADMM

% Problem description must be provided!
problem.n = n; % Number of variables
problem.p = p; % Number of lags
problem.C = C; % Ensure that this is symmetric positive definite



% Options are not mandatory
options.lambda = lambda; % The lambda paramenter for low-rank regularizaion
options.gamma = gamma; % The gamma paramenter that relatively weighs low-rank and sparse regularizaion
options.maxiter = 2000; % Maximum number of iterations
options.rho_max = 1e3; % Maximum rho
options.abstol = 1e-3; % Absolute tolerance
options.reltol = 1e-3; % Relative tolerance
options.compute_relative_duality_gap = false; % Compute the rel. gap at each iteration. This is costly!
options.compute_primal_variables = true; % Compute the primal variables at the end. This is costly!
options.verb = true; % Verbosity


% Initialization not mandatory
x_initial = [];


% Calling the algorithm
[x_sol, infos] = ADMM_sparse_lowrank_AR(problem, x_initial, options);


%% Plots

% Recovery of sparsity pattern
fs = 20;
figure;

subplot(1,2,1), imshow(mat2gray(abs(1-Omega)))
title('Original sparsity pattern','FontSize',fs)
subplot(1,2,2), imshow(mat2gray(abs(1-x_sol.Omega)))
title('Recovered sparsity pattern','FontSize',fs)
ax1 = gca;
set(ax1,'FontSize',fs);


% Evolution of augmented Lagrangian with iterations
fs = 20;
figure;
plot(infos.augmented_Lagrangian,'-','Color','blue','LineWidth',2);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'Number of iterations','FontSize',fs);
ylabel(ax1,'Augmented Lagrangian','FontSize',fs);
title('Evolution of relative duality gap with iterations')

% Evolution of the primal and dual residuals of ADMM with iterations
fs = 20;
g = figure;

subplot(2,1,1);
semilogy(1:length(infos.primal_residual), max(1e-8, infos.primal_residual), 'blue', ...
    1:length(infos.eps_pri), infos.eps_pri, 'k--',  'LineWidth', 2);
ax1 = gca;
set(ax1,'FontSize',fs);
ylabel('||r||_2');
title('Evolution of primal and dual residuals of ADMM')


subplot(2,1,2);
semilogy(1:length(infos.dual_residual), max(1e-8, infos.dual_residual), 'red', ...
    1:length(infos.eps_dual), infos.eps_dual, 'k--', 'LineWidth', 2);
ax1 = gca;
set(ax1,'FontSize',fs);
ylabel(ax1,'||s||_2','FontSize',fs);
xlabel(ax1,'iterations','FontSize',fs);

if options.compute_relative_duality_gap
    % Evolution of relative duality gap with iterations
    fs = 20;
    figure;
    semilogy(infos.relative_duality_gap,'-','Color','blue','LineWidth',2);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Number of iterations','FontSize',fs);
    ylabel(ax1,'Relative duality gap','FontSize',fs);
    title('Evolution of relative duality gap with iterations')
    
    
    % Evolution of the dual cost with iterations
    fs = 20;
    figure;
    plot(infos.cost_dual,'-','Color','blue','LineWidth',2);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Number of iterations','FontSize',fs);
    ylabel(ax1,'Dual cost','FontSize',fs);
    title('Evolution of the dual cost with iterations')
    
    
    % Evolution of the dual cost with time
    fs = 20;
    figure;
    plot(infos.iter_time, infos.cost_dual,'-','Color','blue','LineWidth',2);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time in seconds','FontSize',fs);
    ylabel(ax1,'Dual cost','FontSize',fs);
    title('Evolution of the dual cost with time')

end





