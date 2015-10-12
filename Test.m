% This is a generic test file for calling the ADMM algorithm
% for sparse plus low-rank autoregressive identification
%
% Refer "Sparse plus low-rank autoregressive identification in neuroimaging time series",
% Raphaël Liégeois, Bamdev Mishra, Mattia Zorzi, and Rodolphe Sepulchre,
% Technical report, arXiv:1503.08639, 2015.
%
% This implementation is due to
% Raphaël Liégeois <R.Liegeois@ulg.ac.be> and
% Bamdev Mishra <B.Mishra@ulg.ac.be>, 2015.


clc; clear all; close all



%% Random data

% Choose number of manifest variables
n = 10;

% Choose number of lags
p = 1;


Nt = 500;
X = randn(n, Nt);


% Computation of C from X a for the number of lags that are asked.
temp = xcorr(X', p,'biased'); %If 'unbiased', no guarantee that C is symmetric positive definite!

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
problem.n = n; % Number of manifest variables
problem.p = p; % Number of lags
problem.C = C; % Ensure that this symmetric positive definite


% Options are not mandatory
options.lambda = 0.004; % The lambda paramenter for low-rank regularizaion
options.gamma = 1.25; % The gamma paramenter that relatively weighs low-rank and sparse regularizaion
options.maxiter = 1000; % Maximum number of iterations
options.rho_max = 1e3; % maximum rho
options.abstol = 1e-4; % Absolute tolerance
options.reltol = 1e-4; % Relative tolerance
options.compute_relative_duality_gap = true; % Compute the rel. gap at each iteration. This is costly!
options.compute_primal_variables = true; % Compute the primal variables at the end. This is costly!
options.verb = true; % Verbosity


% Initialization not mandatory
x_initial = [];


% Calling the algorithm
[x_sol, infos] = ADMM_sparse_lowrank_AR(problem, x_initial, options);




%% Plots


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





