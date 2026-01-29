# %%
import numpy as np
import numpy_groupies as npg
import scipy.sparse as sps
import numpy.matlib as npm
import cvxpy as cp
import scipy.io
import networkx as nx


def itril(sz, k = 0, linear_ind = True):
    
    # if isscalar(sz)
        # sz = [sz sz];
    # end
    # m=sz(1);
    # n=sz(2);
    
    # Python implementation
    if np.isscalar(sz):
        sz = [sz, sz]
    m = sz[0]
    n = sz[1]
    
    # Main Diagonal by default
    # if nargin<2
        # k=0;
    # end
    # This is taken care of with default argument k = 0
    
    # nc = n-max(k,0); % number of columns of the triangular part
    
    nc = n - max(k,0)
    
    # lo = ones(nc,1); % lower row indice for each column
    
    lo = [int(i) for i in np.ones(nc)]
    # print('lo is ' + str(lo))
    
    # hi = min((1:nc).'-min(k,0),m); % upper row indice for each column
    
    hi = [min(i - min(k,0), m) for i in range(1,nc+1)]
    
    # print('hi is ' + str(hi))
    # if isempty(lo)
        # I = zeros(0,1);
        # J = zeros(0,1);
    
    if len(lo) == 0: 
        I = []
        J = []
    
    # else
    
    else:
        
        # c=cumsum([0; hi-lo]+1); % cumsum of the length
        
        temp = [1] + [hi[i] - lo[i] + 1 for i in range(len(hi))]
        c = np.cumsum(temp)
        # print('c is ' + str(c))
        
        # I = accumarray(c(1:end-1), (lo-[0; hi(1:end-1)]-1), ... [c(end)-1 1]);
        
        temp = [0] + hi[:len(hi)-1]
        
        # print('temp1 is ' + str(temp))
        
        I = npg.aggregate(c[0:len(c)-1], [lo[i] - temp[i] - 1 for i in range(len(lo))], size = c[len(c)-1])[1:]
        
        # print('first input is ' + str(c[0:len(c)-1]))
        # print('second input is ' + str([lo[i] - temp[i] - 1 for i in range(len(lo))]))
        # print('I after aggregating is ' + str(I))
        
        # I = cumsum(I+1); % row indice
        
        I = np.cumsum([i + 1 for i in I])
        
        # print('I after cummulation is ' + str(I))
        # J = accumarray(c,1);
        
        J = npg.aggregate(c, 1)[1:]; 
        
        # print('J after aggregation is ' + str(J))
        
        # J(1) = 1 + max(k,0); % The row indices starts from this value
        
        J[0] = 1 + max(k,0)
        
        # J = cumsum(J(1:end-1)); % column indice
        
        J = np.cumsum(J[0:len(J)-1])
        
        # print('J after cummulation is ' + str(J))
    # end
    
        
        
    # if nargout<2
    # % convert to linear indices
        # I = sub2ind([m n], I, J);
    # end
    I = [i - 1 for i in I]
    J = [j - 1 for j in J]
    if linear_ind == True:
        I = np.ravel_multi_index([J,I], [m,n])
        return(I)
    return([J,I])


def DuplicationM(n, option = 'lo'):
    """
    if nargin<2
        option = 'lo'; % default
    end
    """
    
    """ 
    if isscalar(n)
        n = [n n];
    end
    """
    if np.isscalar(n):
        n = [n, n]
    
    """
    switch lower(option(1))
        case 'l' % u, lo, LO, LOWER ...
            [I J] = itril(n);
        case 'u' % u, up, UP, UPPER ...
            [I J] = itriu(n);
        otherwise
            error('option must be ''lo'' or ''up''');
    end
    """
    if option[0].lower() == 'l':
        I, J = itril(n,0,False)
    elif option[0].lower() == 'u':
        J, I = itril(n,0,False)
    else:
        # print("Error, optioin mus be 'lo' or 'up'.")
        return()
    
    I = [x for _, x in sorted(zip(J, I))]
    J = sorted(J)
    """
    % Find the sub/sup diagonal part that can flip to other side
    loctri = find(I~=J & J<=n(1) & I<=n(2));
    """
    # print('I is ' + str(I))
    # print('J is ' + str(J))
    loctri = [i for i in range(len(I)) if 
              (I[i] != J[i]) and 
              (J[i] <= n[0]-1) and 
              (I[i] <= n[1]-1)]
    # print('loctri = ' + str(loctri))
    
    """
    % Indices of the flipped part
    Itransposed = sub2ind(n, J(loctri), I(loctri));
    """
    
    arg1 = [J[i] for i in loctri]
    arg2 = [I[i] for i in loctri]
    
    Itransposed = np.ravel_multi_index([arg1, arg2], n)
    # print('Itransposed = ' + str(Itransposed))
    """
    % Convert to linear indice
    I =  sub2ind(n, I, J);
    """
    I = np.ravel_multi_index([I, J], n)
    
    """
    % Result
    M = sparse([I; Itransposed], ...
               [(1:length(I))'; loctri], 1, prod(n), length(I));
    """
    arg1 = np.append(I,Itransposed)
    arg2 = np.append([i for i in range(len(I))], loctri)
    d = [1]*len(arg1)
    
    # print('length of d is ' + str(len(d)))
    # print('length of I is ' + str(len(I)))
    # print('length of Itransposed is ' + str(len(Itransposed)))
    M = sps.csr_matrix((d, (arg1, arg2)), shape = (np.prod(n),len(I)))
    return(M)


def laplacian_constraint_vech(N):
    """    
    %% matrix for objective (vech -> vec)
    mat_obj = DuplicationM(N);
    """
    mat_obj = DuplicationM(N)
    
    # Not complete
    """
    X = ones(N);
    [r,c] = size(X);
    i     = 1:numel(X);
    j     = repmat(1:c,r,1);
    B     = sparse(i',j(:),X(:))';
    mat_cons1 = B*mat_obj;
    """
    X = np.ones([N,N])
    r, c = X.shape
    i = range(r*c)
    j = np.matlib.repmat(range(c), r, 1)
    B = sps.csr_matrix((X.flatten('F'), (i, j.flatten('F'))), dtype = np.int_)
    B = B.transpose()
    mat_cons1 = B@mat_obj
    """    
    %% matrix for constraint 2 (non-positive off-diagonal entries)
    for i = 1:N
        tmp{i} = ones(1,N+1-i);
        tmp{i}(1) = 0;
    end
    mat_cons2 = spdiags(horzcat(tmp{:})',0,N*(N+1)/2,N*(N+1)/2);
    """
    Tmp = []
    for i in range(N):
        tmp = [1]*(N-i)
        tmp[0] = 0
        Tmp = Tmp + tmp
    mat_cons2 = sps.spdiags(Tmp, 0, N*(N+1)//2, N*(N+1)//2)
    
    
    """    
    %% vector for constraint 3 (trace constraint)
    vec_cons3 = sparse(ones(1,N*(N+1)/2)-horzcat(tmp{:}));
    """
    
    arg1 = [1 - i for i in Tmp]
    vec_cons3 = sps.csr_matrix(arg1, dtype = np.int_)
    
    
    """    
    %% create constraint matrices
    % equality constraint A2*vech(L)==b2
    A1 = [mat_cons1;vec_cons3];
    b1 = [sparse(N,1);N];
    """
    
    A1 = sps.vstack([mat_cons1, vec_cons3])
    b1 = sps.vstack([sps.csr_matrix((N,1), dtype = np.int_), 
                     sps.csr_matrix(([N], ([0],[0])))])
    
    
    """
    % inequality constraint A1*vech(L)<=b1
    A2 = mat_cons2;
    b2 = sparse(N*(N+1)/2,1);
    """
    
    A2 = mat_cons2
    b2 = sps.csr_matrix((N*(N+1)//2, 1), dtype = np.int_)
    
    return([A1,b1,A2,b2,mat_obj])
    

def optimize_laplacian_gaussian(N,Y,alpha,beta):

    """    
    %% Laplacian constraints
    [A1,b1,A2,b2,mat_obj] = laplacian_constraint_vech(N);
    p = vec(Y*Y')';
    """
    A1, b1, A2, b2, mat_obj = laplacian_constraint_vech(N)
    p = (Y@np.transpose(Y)).flatten('F')
    
    L = cp.Variable((N*(N+1)//2, 1))
    constraints = [A1@L == b1,
                   A2@L <= b2]
    objective = cp.Minimize(alpha*(p@mat_obj@L) + beta*cp.sum_squares(mat_obj@L))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    EL = np.reshape(mat_obj@(L.value), (N,N))
    return(EL)

   

"""
function [L,Y,L_harvard] = graph_learning_gaussian(X_noisy,param)
% Learning graphs (Laplacian) from structured signals
% Signals X follow Gaussian assumption
"""
def graph_learning_gaussian(X_noisy, param):
    """
    N = param.N;
    max_iter = param.max_iter;
    alpha = param.alpha;
    beta = param.beta;
    """
    N = param['N']
    max_iter = param['max_iter']
    alpha = param['alpha']
    beta = param['beta']

    """
    objective = zeros(max_iter,1);
    Y_0 = X_noisy;
    Y = Y_0;
    """
    objective = [0]*max_iter
    Y_0 = X_noisy.copy()
    Y = Y_0.copy()     # changed to deep copies 07-04-2022
    
    
    for i in range(max_iter):
        
        # Step 1: given Y, update L
        L = optimize_laplacian_gaussian(N,Y,alpha,beta)
        
        # Step 2: given L, update Y
        R = np.linalg.cholesky(np.identity(N) + alpha*L)
        Rt = np.transpose(R)
        arg1 = np.linalg.lstsq(Rt, Y_0, rcond=None)[0]   # add rcond=None to suppress the warning message
        #print('arg1 shape is ' + str(arg1.shape))
        #print('R shape is ' + str(R.shape))
        Y = np.linalg.lstsq(R, arg1, rcond=None)[0]
        
        # Store objective
        arg1 = np.linalg.norm(Y-Y_0, 'fro')**2 
        #print("arg1 is: ", arg1)
        arg2 = alpha*(np.transpose((Y@np.transpose(Y)).flatten('F'))@(L.flatten('F')))
        #print("arg2 is: ", arg2)
        arg3 = beta*np.linalg.norm(L, 'fro')**2
        #print("arg3 is: ", arg3)
        objective[i] = arg1 + arg2 + arg3
        #print("Print the objective at the iteration ", i, ": ", objective[i])
        # Stopping criteria
        if i>=2 and abs(objective[i] - objective[i-1]) < 10**(-4):
            break
        
    # return is changd to outside of the for loop, as the final return of the funciton 07-04-2022    
    return([L, Y])
