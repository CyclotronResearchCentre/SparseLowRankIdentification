function [x_sol, infos, options] = ADMM_sparse_lowrank_AR(problem, x, options)
    % ADMM algorithm to solve the sparse plus low-rank autoregressive
    % identification problem.
    %
    % function [x_sol, infos, options] = ADMM_sparse_lowrank_AR(problem, x, options)
    %
    %
    % Parementers
    % -----------
    % problem: Stuctured array containing the problem description like
    %          sample covariance matrix (C), number of manifest variables (n), and number of lags (p).
    %
    % x      : Stuctured array of variables used in optimization.
    %
    % x.Z    : a matrix of dimension (n, n,(p+1))
    % x.Y    : a symmetric matrix of dimension (n(p+1), n(p+1))
    % x.M    : a symmetric matrix of dimension (n(p+1), n(p+1))
    %
    % options: Structured array with options for the algorithm
    %
    % options.lambda : The lambda paramenter for low-rank
    %                  regularizaion
    % options.gamma :  The gamma paramenter that relatively weighs low-rank
    %                  and sparse regularizaion
    % options.maxiter: Maximum number of iterations
    % options.rho_max: Maximum rho
    % options.rho:     Initial rho
    % options.abstol:  Absolute tolerance, as defined in the paper
    % options.reltol:  Relative tolerance, as defined in the paper
    % options.reltol_cost: Relative tolerace for the change in augmented
    %                      Lagrangian
    % options.compute_relative_duality_gap:  Rel. gap at each iteration
    % options.compute_primal_variables:      Compute the primal variables at the end
    % options.verb:                          Verbosity
    % options.sigma_armijo:   The constant factor used in the Armijo linesearch
    % options.stepsize:       Initial stepsize guess for backtracking linesearch
    % options.ls_maxiter:     Maximum of linesearch allowed
    %
    %
    % Output:
    % -------
    % x_sol: Structured array with the final iterates Z, Y, M, and also...
    %
    % x_sol.h:      The number of latent variables
    % x_sol.L:      The low-rank part
    % x_sool.Omega: The sparsity pattern
    % x_sol.S:      The sparse part
    % x_sol.Delta:  The sum of sparse plus low-rank parts
    %
    % infos: Structured array with additional information
    % options: Structured array with options used in the algorithm
    
    % Refer "Sparse plus low-rank autoregressive identification in neuroimaging time series",
    % Raphaël Liégeois, Bamdev Mishra, Mattia Zorzi, and Rodolphe Sepulchre,
    % Technical report, arXiv:1503.08639, 2015.
    %
    % This implementation is due to
    % Raphaël Liégeois <R.Liegeois@ulg.ac.be> and
    % Bamdev Mishra <b.mishra@ulg.ac.be>, 2015.
    
    
    
    
    
    %% Set default parameters
    if ~isfield(options,'lambda'); warning('Problem:incomplete_information', 'lambda not supplied. Setting it to 1. \n'); options.lambda = 1; end
    if ~isfield(options,'gamma'); warning('Problem:incomplete_information','gamma not supplied. Setting it to 1. \n'); options.gamma = 1; end
    if ~isfield(options,'maxiter'); options.maxiter = 1000; end % Maximum number of iterations
    if ~isfield(options,'rho_max'); options.rho_max = 1e3; end % maximum rho
    if ~isfield(options,'rho'); options.rho = 1; end % Initial rho
    if ~isfield(options,'abstol'); options.abstol = 1e-4; end % Absolute tolerance
    if ~isfield(options,'reltol'); options.reltol = 1e-4; end % Relative tolerance
    if ~isfield(options,'reltol_cost'); options.reltol_cost = 1e-12; end % Relative tolerance for cost variation
    if ~isfield(options,'compute_relative_duality_gap'); options.compute_relative_duality_gap = true; end %  Compute the rel. gap at each iteration
    if ~isfield(options,'compute_primal_variables'); options.compute_primal_variables = true; end % Compute the primal variables at the end
    if ~isfield(options,'verb'); options.verb = true; end % Verbosity
    
    if ~isfield(options,'sigma_armijo'); options.sigma_armijo = 0.5; end % For linesearch
    if ~isfield(options,'stepsize'); options.stepsize = 1; end % Initial stepsize guess
    if ~isfield(options,'ls_maxiter'); options.ls_maxiter = 50; end % Maximum number of line-search steps
    
    if ~isfield(options,'rho_inc'); options.rho_inc = 2; end % rho increment for a particular updating of rho (if asked)
    if ~isfield(options,'rho_dec'); options.rho_dec = 2; end % rho deccrement for a particular updating of rho (if asked)
    if ~isfield(options,'mu'); options.mu = 10; end % for a particular updating of rho (if asked)
    
    
    %% Set problem description
    if ~isfield(problem,'C'); error('Error. "C" covariance matrix must be given.'); end
    if ~isfield(problem,'n'); error('Error. "n" corresponding to number of variables must be given.'); end
    if ~isfield(problem,'p'); error('Error. "p" corresponding to number of lags must be given.'); end
    
    n = problem.n; % Should be equal to size(C, 1)
    p = problem.p;
    
    
    
    
    %% Parameters
    
    sigma_armijo = options.sigma_armijo;% For sufficient decrease condition in Armijo linesearch
    contraction_factor = 0.5;% Stepsize update
    stepsize = options.stepsize;% Starting stepsize
    maxiter = options.maxiter;% total number of iterations
    ls_maxiter = options.ls_maxiter;
    rho_max = options.rho_max; % maximum value of rho
    rho = options.rho;% Regularization parameter in ADMM
    rho_inc = options.rho_inc;% Only for a particular update of rho
    rho_dec = options.rho_dec;% Only for a particular update of rho
    mu = options.mu;% Only for a particular update of rho
    abstol   = options.abstol; % Absolute tolerace defined in the paper. It makes sense only for a fixed rho.
    reltol   = options.reltol; % Absolute tolerace defined in the paper. It makes sense only for a fixed rho.
    reltol_cost = options.reltol_cost; % A stopping criterion. It makes sense only for a fixed rho.
    lambda = options.lambda;
    lambdaI = lambda*speye(n*(p+1));
    options.lambdaI = lambdaI; % Create an additional field that is used often in the code
    
    
    %% Initializaiton
    % Initialization of the variables Z,Y and M if not supplied by th user
    
    if ~all(isfield(x,{'Z','Y','M'}) == 1)
        Z = zeros(n,n,p+1);
        T_Z = get_T(Z);
        Y = eye(n*(p+1));
        M = zeros(n*(p+1));
        
        x.Z = Z;
        x.T_Z = T_Z;
        x.Y = Y;
        x.M = M;
        x.W = get_W(problem, x);
    end
    
    
    %% Information structure
    infos.equality_constraint = [];
    infos.positive_semidefiniteness_constraint = [];
    infos.primal_residual = [];
    infos.dual_residual = [];
    infos.eps_pri = [];
    infos.eps_dual = [];
    infos.augmented_Lagrangian = [];
    infos.iter_time = [];
    infos.cost_dual = [];
    infos.cost_primal = [];
    infos.relative_duality_gap = [];
    infos.rho = [];
    
    
    
    %% Main Code
    t_begin = tic();
    
    for iter = 1 : maxiter,
        Z = x.Z;
        Y = x.Y;
        M = x.M;
        W = x.W;
        options.rho = rho; % Add an extra field to the options structure
        
        
        %% Compute the cost (the augmented Lagrangian) at the current iterate.
        cost_old = cost(problem, x, options); % Old cost does not change with stepsize t.
        
        
        %% First update for for the variable Z
        
        % Alpha is the gradient of the augmented Lagrangian.
        alpha = grad(problem, x, options); %
        long_grad_Z = zeros(n, n*(p+1));
        for i = 1 : p+1
            long_grad_Z(:, (i-1)*n + 1 : i*n) = alpha(:,:,i);
        end
        
        
        % Computing the Gradient map
        Z_new = proj_C1(Z - stepsize*alpha, options); %
        Gm = (1/stepsize)*(Z - Z_new); % The Gradient map.
        long_grad_map = zeros(n, n*(p+1));
        for i = 1 : p+1
            long_grad_map(:,(i-1)*n+1:i*n) = Gm(:,:,i);
        end
        T_Z_new = get_T(Z_new);
        
        x_new.Z = Z_new;
        x_new.T_Z = T_Z_new;
        x_new.Y = Y;
        x_new.M = M;
        
        % The new cost
        cost_new = cost(problem, x_new, options); % Next cost
        generalized_gradient_norm_sq = (long_grad_Z(:)'*long_grad_map(:)); % Generalized norm square of the gradient
        
        boolean_positive_definiteness = check_positive_definiteness(problem, x_new); % Positive definiteness condition
        boolean_sufficient_decrease = cost_new <= cost_old -  stepsize*sigma_armijo*generalized_gradient_norm_sq;% % Sufficient decrease condition
        
        
        % Backtracking linesearch over the constraint C1 to find the next update of Z that
        % satisfies the positive definiteness and sufficient decrease
        % conditions.
        g = 1;
        stop = false;
        while ~(boolean_positive_definiteness && boolean_sufficient_decrease)
            
            stepsize = stepsize*contraction_factor;
            g = g + 1;
            
            % Computing the Gradient map
            Z_new = proj_C1(Z - stepsize*alpha, options);
            Gm = (1/stepsize)*(Z - Z_new);
            for i = 1 : p+1
                long_grad_map(:,(i-1)*n+1:i*n) = Gm(:,:,i);
            end
            
            T_Z_new = get_T(Z_new);
            x_new.Z = Z_new; % update Z only
            x_new.T_Z = T_Z_new; % update Z only
            cost_new = cost(problem, x_new, options); % Next cost
            generalized_gradient_norm_sq = (long_grad_Z(:)'*long_grad_map(:)); % Generalized norm square of the gradient
            
            boolean_positive_definiteness = check_positive_definiteness(problem, x_new); % Positive definiteness condition
            boolean_sufficient_decrease = cost_new <= cost_old -  stepsize*sigma_armijo*generalized_gradient_norm_sq; % Sufficient decrease condition
            
            if g > ls_maxiter
                if options.verb,
                    fprintf('\nArmijo condition never satisfied while updating Z... Stepsize is %3.3e.  \n', stepsize);
                end
                stop = true;
                break
            end
        end
        W_new = get_W(problem, x_new);
        x_new.W = W_new;
        infos.augmented_Lagrangian = [infos.augmented_Lagrangian; cost_new];
        if stop
            break
        end
        
        %         % Additional stopping criterion based on the variation of the cost for
        %         % minimizing the variable Z
        %         if abs(cost_new - cost_old) / abs(cost_old) < reltol_cost && rho == rho_max
        %             if options.verb,
        %                 fprintf('\nThe augmented Lagrangian does not vary that much. \n');
        %             end
        %             break;
        %         end
        
        
        
        %% Second update for the variable Y
        % we have the closed form solution
        Y_new = proj_C2((M/rho) + T_Z_new + lambdaI); 
        x_new.Y = Y_new;
        
        
        %% Third update for the dual varible M
        % we have a gradient ascent update
        M_new = M - rho*(Y_new - T_Z_new - lambdaI); 
        x_new.M = M_new;
        
        
        %% Computing and collecting informtation
        
        equality_constraint = norm(Y_new - T_Z_new - lambdaI,'fro');
        positive_semidefiniteness_constraint = min(eig(Y_new)); % Minimum eig value
        
        % Output information
        if options.compute_relative_duality_gap
            
            % Computing the primal variables and cost of the original optimization
            % problem
            [cost_primal, cost_dual, ~] = compute_costs_and_primal_variables(problem, x_new, options); % Computing the primal variables. This is costly!
            
            
            % Relative duality gap computation
            d_gap = abs(cost_primal - cost_dual) / abs(cost_primal); % Relative duality gap
            
            
            if options.verb,
                fprintf('[%0.4d] Augmented Lagrangian = %7.3e, Dual cost = %7.3e, stepsize = %3.3e, rho = %3.3e, rel. duality gap = %7.3e, equality constraint = %7.3e, symm. pos. def. constraint = %7.3e  \n', iter, cost_new, cost_dual, stepsize, rho,d_gap,equality_constraint, positive_semidefiniteness_constraint);
                
            end
            infos.cost_dual = [infos.cost_dual; cost_dual];
            infos.cost_primal = [infos.cost_primal; cost_primal];
            infos.relative_duality_gap = [infos.relative_duality_gap;  d_gap];
            
        else
            
            
            if options.verb
                fprintf('[%0.4d] Augmented Lagrangian = %7.3e, stepsize = %3.3e, rho = %3.3e\n', iter, cost_new, stepsize, rho);
                
            end
        end
        
        % Collect information
        infos.equality_constraint = [infos.equality_constraint; equality_constraint];
        infos.positive_semidefiniteness_constraint = [infos.positive_semidefiniteness_constraint; abs(positive_semidefiniteness_constraint)];
        infos.iter_time = [infos.iter_time; toc(t_begin)];
        infos.rho = [infos.rho; rho]; % We update rho at each iteration.
        
        
        
        %% Standard stopping criteria based on primal and dual residual error
        primal_residual = equality_constraint; % As defined in the paper.
        infos.primal_residual = [infos.primal_residual; primal_residual];
        
        Y_diff = (Y_new - Y);
        dual_residual = rho*norm(get_D(Y_diff, n , p), 'fro'); % As defined in the paper.
        infos.dual_residual = [infos.dual_residual; dual_residual];
        
        infos.eps_pri = [infos.eps_pri; sqrt(n*(p+1)*n*(p+1))*abstol + reltol*max([lambda*sqrt(n*(p+1)); norm(T_Z_new,'fro'); norm(Y_new,'fro')]) ];
        infos.eps_dual = [infos.eps_dual; sqrt(n*(p+1)*n)*abstol + reltol*norm(get_D(M_new,n,p),'fro')];
        
        % Stopping criterion 
        if primal_residual < infos.eps_pri(end) && ...
                dual_residual < infos.eps_dual(end) && ...
                rho == rho_max
            if options.verb,
                fprintf('\nPrimal and dual residual sufficiently decreased. \n');
            end
            break;
        end
        
        
        
        %% Update rho and t
        
        % Adaptive stepisize strategy to guess the stepsize for the next
        % iteration.
        stepsize = stepsize / contraction_factor;
        
        % Preferred choice for updating rho.
        rho = min(rho_max, 1.1*rho); % Boyd's suggestion
        
        %             % Second choice from Stephen Boyd's book
        %             if primal_residual > mu*dual_residual,
        %                 rho = rho_inc*rho; % Increase rho
        %             elseif dual_residual > mu*primal_residual
        %                 rho = rho/rho_dec; % Decrease rho
        %             end
        
        
        %% Update of the variables
        
        x = x_new;
    end
    
    %%
    if iter == maxiter,
        if options.verb,
            fprintf('\nMaximum number of iterations exhausted. \n');
        end
    end
    
    
    x_sol = x_new;
    
    if options.compute_primal_variables
        % Primal and dual cost computation
        [cost_primal, cost_dual, y] = compute_costs_and_primal_variables(problem, x_new, options); % Computing the primal variables. This is costly!
        
        % Relative duality gap computation
        d_gap = abs(cost_primal - cost_dual) / abs(cost_primal); % Relative duality gap
        
        infos.cost_dual_final = cost_dual;
        infos.cost_primal_final = cost_primal;
        infos.relative_duality_gap_final = d_gap;
        
        
        
        % Collect the primal variables
        x_sol.L = y.L;
        x_sol.Omega = y.Omega; % The sparsity pattern
        x_sol.h = y.h; % The low-rank part
        x_sol.Delta = y. Delta; % The sum of sparse plus low-rank parts
        x_sol.S = y.S; % The sparse part
        
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   FUNCTIONS USED IN MAIN CODE   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Checking one of the conditions for the backtracking step
function boolean1 = check_positive_definiteness(problem, x)
    
    C = problem.C;
    Z = x.Z;
    T_Z = x.T_Z;
    
    
    n = size(Z, 1);
    
    V = C + T_Z;
    V = symm(V);
    
    % First part, positive semidefiniteness of V
    cond11 = min(eig(V)) >= -1e-8; % BM: earlier it was 0
    
    % Second part, positive semidefiniteness of the Schur complement
    V00 = V(1:n, 1:n);
    V1p0 = V(n+1 : end, 1:n);
    V1p1p = V(n+1 : end, n+1 : end);
    
    V_temp = V00 - V1p0'*(V1p1p\V1p0);
    V_temp = symm(V_temp); % BM
    cond12 = min(eig(V_temp))>= -1e-8; % BM: earlier it was 0
    
    % Synthesis
    boolean1 = cond12 && cond11;
end



%% Function to compute the augmented Lagrangian
function val = cost(problem, x, options)
    val = phi(problem, x) + trace(x.M(:)'*x.T_Z(:)) + (options.rho/2)*norm(x.Y - x.T_Z - options.lambdaI,'fro')^2;
end



%% Function to compute the gradient of the augmented Lagrangian
function grad_Z = grad(problem, x, options)%x is an element of M^{n*p}
    C = problem.C;
    Z = x.Z;
    T_Z = x.T_Z;
    Y = x.Y;
    M = x.M;
    rho = options.rho;
    
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    lambdaI = options.lambdaI;
    
    grad_Z = grad_phi(C, Z, T_Z) +  get_D_short(M + rho*(T_Z + lambdaI - Y), n, p);
    
end



%% Function to evaluate phi, defined in the paper
function r = phi(problem, x)
    C = problem.C;
    Z = x.Z;
    T_Z = x.T_Z;
    
    n = size(Z,1);
    
    V = C + T_Z;
    
    V00 = V(1:n,1:n);
    V1p0 = V(n+1:end,1:n);
    V1p1p = V(n+1:end,n+1:end);
    V_temp = V00 - V1p0'*(V1p1p\V1p0);
    V_temp = symm(V_temp);
    r = -log(det(V_temp)) - n;
end



%% Function to compute the gradient of phi
function grad_Z = grad_phi(C, Z, T_Z)%x is an element of M^{n*p}
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    
    Y = sparse([zeros(n*p,n) eye(n*p)]);
    
    V_temp = C + T_Z;
    
    T1 = speye(size(V_temp))/V_temp;
    T2Y = ((Y*V_temp)*Y')\Y;  % T2 = inv((Y*V_temp)*Y');
    
    grad_Z1 = get_D(-T1 + Y'*T2Y, n, p);
    grad_Z = zeros(n, n, p+1);
    
    for i = 1 : p+1
        grad_Z(:,:,i) = grad_Z1(:,(i-1)*n + 1 : i*n);
    end
end



%% Function to project onto the C1 constraint
function Z_projected = proj_C1(Z, options) %Z is the U of the equations p.2699
    gamma = options.gamma;
    lambda = options.lambda;
    
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    X = zeros(1, 2*(p+1)); % A row vector
    gammalambda = gamma*lambda;
    
    Z_projected = zeros(size(Z));
    
    Z(:,:,1) = symm(Z(:, :, 1)); % Ensure that the first block is symmetric
    
    for i = 1 : n-1
        
        for j = (i+1) : n % For each element we perform LASSO
            % Reformulation using vectors
            X(1 : p+1) = Z(i,j,:);
            X(p+2 : end) = Z(j,i,:);
            
            absX = abs(X);
            Z_temp = projectSortC(absX', gammalambda); % Using the mex file from EWOUT VAN DEN BERG, MARY SCHMIDT, MICHAEL P. FRIEDLANDER, AND YEVIN MURPHY
            Z_temp = (sign(X').*Z_temp)';
            
            Z_projected(i,j,:) = Z_temp(1 : p+1);
            Z_projected(j,i,:) = Z_temp(p+2 : end);
            
        end
        
    end
    
    Z_projected(:,:,1) = symm(Z_projected(:, :, 1)); % Ensure that the first block is symmetric
    
    
end



%%  Function to project onto the cone of positive definite matrices
function Y_truncated = proj_C2(Y)
    %Z_mat is of size(n*(p+1),n*(p+1))
    Y = symm(Y);
    [V, D] = eig(Y); %D contains eigenvalues; Columns of V contain corresponding eigenvectors
    d = diag(D);
    w1 = ones(size(V,1),1);
    w1(d<=0)=0;
    w = logical(w1);
    V1 = V(:,w);
    d1 = d(w);
    Y_truncated = symm((V1*diag(d1))*V1');
    
end




%% Function to compute T(Z), outputs a block Toeplitz matrix
function T = get_T(Z)
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    
    
    T = zeros(n*(p+1));
    for i=1:p+1
        for j=1:p+1
            if i<j
                T(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=Z(:,:,(j-i)+1);
            else
                T(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=(Z(:,:,(i-j)+1))';
            end
        end
    end
    
    T = symm(T); % It should be symmetric
end


%% Function to compute the adjoint of T with output in 3-d array
function D = get_D_short(X, n, p)
    D = zeros(n,n,(p+1));
    %D0
    for l=0:p
        D(:,:,1)=D(:,:,1)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
    end
    %Dj for j in [1,p]
    for m=1:p
        for l=0:p-m
            D(:,:,m+1) = D(:,:,m+1) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
        end
    end
    
    
end


%% Function to compute the adjoint of T with output in a matrix of size n x (p+1)
function D = get_D(X, n, p)
    
    D = zeros(n,n*(p+1));
    %D0
    for l=0:p
        D(:,1:n)=D(:,1:n)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
    end
    
    %Dj for j in [1,p]
    for m=1:p
        for l=0:p-m
            D(:,(m*n+1:(m+1)*n)) = D(:,(m*n+1:(m+1)*n)) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
        end
    end
end


%% Function to compute W
function W = get_W(problem, x)
    C = problem.C;
    Z = x.Z;
    T_Z = x.T_Z;
    
    
    n = size(Z,1);
    
    %     V = get_V(C, Z);
    V = C + T_Z;
    
    V00 = V(1:n,1:n);
    V1p0 = V(n+1:end,1:n);
    V1p1p = V(n+1:end,n+1:end);
    W = V00 -V1p0'*(V1p1p\V1p0);
    W = symm(W);
end


%% Function to compute the symmetric part of a matrix
function D = symm(D)
    D = 0.5*(D + D');
end



%% Function to extract the first block of a matix of size n(p+1) x n(p+1)
function B = get_first_block(A,n_var)
    n = n_var;
    B = A(1:n,1:n);
end




%% Function to compute the "infinity norm".
function h_BM = get_h(D)
    % D is in a 3d format.
    
    Dabs = abs(D);
    Dabs_max_matrix = max(Dabs, [], 3);
    Dabs_max_matrix_offdiag = Dabs_max_matrix - diag(diag(Dabs_max_matrix)); % Remove the diagonal
    DU = triu(Dabs_max_matrix_offdiag); % Upper triangular part
    DL = tril(Dabs_max_matrix_offdiag); % Lower triangular part
    DLt = DL'; % Transpose
    h_BM = sum(max([DLt(:), DU(:)], [], 2));
    
end


%%
function S = get_short_V(V, n, p, n_latent)
    l = n_latent; % number of latent variables.
    S=zeros(n,l,(p+1));
    for i=1:p+1
        S(:,:,i)=V((i-1)*n+1:i*n,:);
    end
end



%% Compute the primal variables and the primal and dual costs
function [Cost_p, Cost_d, y] = compute_costs_and_primal_variables(problem, x, options)
    
    C = problem.C;
    n = problem.n;
    p = problem.p;
    
    Z = x.Z;
    T_Z = x.T_Z;
    W = x.W;
    
    gamma = options.gamma;
    lambda = options.lambda;
    
    %% Delta
    R = C;
    
    
    Delta = (R + T_Z)\(sparse([W, zeros(n,n*p); zeros(n*p,n), zeros(n*p,n*p) ])/(R + T_Z));
    Delta = 0.5*(Delta + Delta'); % numerical accuracy
    
    
    %% Restriction to Sigma
    [U D] = eig(lambda*eye(n*(p+1)) + T_Z);
    d = diag(D);
    w1 = zeros(size(U,1),1);
    w1(abs(d) <= 1e-6) = 1; % RL: changed value to 1e-6 in order to have same h in cvx and admm
    w = logical(w1);
    V = U(:,w);
    h = size(V, 2);
    
    
    %% Sparsity pattern Omega
    Omega = ones(n,n);
    Zabs = abs(Z);
    Zabs_matrix = sum(Zabs, 3);
    Zabs_matrix = Zabs_matrix + Zabs_matrix' + diag(1 + lambda*gamma*ones(n,1)); % BM: Ignoring the diagonal
    Omega(Zabs_matrix  - lambda*gamma < -1e-8) = 0;
    
    [I J] = find(Omega == 0); % Collect the indices of the sparsity pattern
    
    my_X = get_D_short(Delta, n, p);
    
    
    %% Computing the low-rank and sparse parts
    H_dummy = ones(h,h);
    [IH, JH] = find(triu(H_dummy) == 1); % Indices of the upper triangular part of a symmetric matrix
    IJH = sub2ind(size(H_dummy), IH, JH);
    
    
    if h > 0
        my_V = get_short_V(V,n,p,h);
        
        rhs = zeros((p)*length(I) + length(I)/2, 1);
        
        %         lhs = zeros((p)*length(I) + length(I)/2, h^2); % BM: If taking the full size
        lhs = zeros((p)*length(I) + length(I)/2, h*(h+1)/2); % BM: If taking the half size
        
        count = 0;
        Hzeros = zeros(h, h);
        
        for z = 0 : p % covering all the D operations
            
            for ii = 1 : length(I) % same as the length of J
                k = I(ii); % ith index
                l = J(ii); % jth index
                
                %computing left hand side term
                sumkl = Hzeros;
                for y = 0 : (p-z)
                    
                    left_vector = my_V(k, :, y + 1);
                    right_vector = my_V(l, :, y + 1 + z);
                    
                    if z==0
                        sumkl = sumkl  +  left_vector'*right_vector;
                    else
                        sumkl = sumkl  +  2*left_vector'*right_vector;
                    end
                    
                end
                
                sumkl = (sumkl + sumkl')/2; %Taking symmetric part of first term
                % in the scalar product in order
                % to have symmetric H
                sumklvec_full = sumkl(:);
                
                sumklvec = sumklvec_full(IJH); % BM: Half size
                
                if z == 0
                    if k < l
                        count = count + 1;
                        rhs(count, :) =  -my_X(k,l,z+1);
                        lhs(count, :) = sumklvec';
                    end
                else
                    count = count + 1;
                    rhs(count, :) = -my_X(k,l,z+1);
                    lhs(count, :) = sumklvec';
                    
                end
                
            end
            
            
        end
        
        if sum(Omega(:) == 0) > 0 % It means that there are some active equations to be solved
            
            X_sol = lhs\sparse(rhs); % This is indeed the most computationally expensive step
            
            % If taking only half the size
            H = zeros(h, h);
            for jj = 1 : length(IH);
                if IH(jj) == JH(jj)
                    H(IH(jj), JH(jj)) = X_sol(jj);
                else
                    H(IH(jj), JH(jj)) = X_sol(jj)/2; % BM: Half because of the adjoint operation; manually it should be clear.
                    H(JH(jj), IH(jj)) = X_sol(jj)/2;
                end
            end
            
            H  = symm(H); % Numerical puropose.
            
            [U_H, D_H] = eig(H); % This is a small eigenvalue decomposition, should be okay.
            d_H = diag(D_H);
            
            
            if min(d_H) > 0
                if options.verb,
                    if norm(lhs*X_sol - rhs, 'fro')/norm(rhs, 'fro') <1e-6
                        fprintf('Accurate solution to linear system for computing low-rank part. \n');
                    else
                        fprintf('Computed low-rank part is Positive Definite but inacurate solution to linear system. \n');
                    end
                end
            else
                if options.verb,
                    fprintf('No accurate solution to linear system but creating a proper low-rank candidate. \n');
                end
                w1 = ones(h,1);
                w1(d_H <= 0) = 0;
                w = logical(w1);
                U_H1 = U_H(:,w);
                d_H1 = d_H(w);
                H = U_H1*diag(d_H1)*U_H1';
                H  = 0.5*(H + H');
            end
        else
            % Omega is all ones, i.e., there are no equations to be solved
            % equivalently, H can be not well defined.
            H = eye(h);
            
        end
        
        % construction of L and S
        L = (V*H)*V';
        L = symm(L); % Low-rank part
        S = Delta + L; % Sparse part
        
    else % This corresponds to the case h == 0.
        H = 0;
        L = zeros(n*(p+1),n*(p+1)); % Low rank part
        S = Delta; % Sparse part
    end
    
    %% Projecting the sparse part onto the sparsity pattern
    
    S_short = get_D_short(S, n ,p); % Get the 3-d array
    for jj = 1:p+1 % small loop, affordable
        S_short(:, :, jj) = Omega.*S_short(:, :, jj); % Project back the sparsity pattern
    end
    
    
    %% Primal and dual cost evaluation
    
    term1 = -log(det(get_first_block(Delta,n)));
    term2 = (C(:)'*Delta(:));
    term3 = (lambda*gamma)*get_h(S_short);
    term4 = lambda*trace(H); % lambda*trace(L);
    
    Cost_p = term1 + term2 + term3 + term4;
    Cost_d = log(det(W)) + n;
    
    %% Collect information
    y = [];
    y.Omega = Omega;
    y.h = h; % Number of latent variables
    y.L = L; % The low-rank part
    y.Delta = Delta; % The sum of sparse plus low-rank parts
    y.S = S_short; % The sparse part
    
end


