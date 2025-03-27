import torch

def PISSA(W, r):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    S_root = torch.sqrt(S_r)
        
    U_mod = U_r @ torch.diag_embed(S_root)
    V_mod = torch.diag_embed(S_root) @ Vh_r
    
    return U_mod, V_mod

def CORDA_ORIGINAL(W, X, r):
    X = X.to(W.device)
    C = X @ X.T
    C = C.float()
    
    matrix = W @ C
    if not torch.isfinite(matrix).all():
        print("Матрица содержит не-конечные значения")

    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        
    U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    Vt = Vt.T
        
    A = U[:, :r]
    
    # C_inv = torch.linalg.inv(C)
    
    I = torch.eye(C.size(0)).to(W.device)
    C_inv = torch.linalg.lstsq(C, I).solution
    C_inv = torch.nan_to_num(C_inv, nan=0.0, posinf=1.0, neginf=-1.0)

    B = torch.diag_embed(S[:r]) @ (Vt[:r, :] @ C_inv)
    
    U, S, VT = torch.linalg.svd(B, full_matrices=False)
    A = A @ U @ torch.diag_embed(torch.sqrt(S))
    B = torch.diag_embed(torch.sqrt(S)) @ VT
    
    X = X.to('cpu')
    return A, B

def CORDA(W, X, r):
    X = X.to(W.device)
    C = X @ X.T
    C = C.float()
    
    matrix = W @ C
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        
    U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    Vt = Vt.T
    
    A = U[:, :r]
    B = U[:, :r].T @ W
    
    T, S, Vt = torch.linalg.svd(B, full_matrices=False)
    S = torch.sqrt(S)
    A = A @ T @ torch.diag_embed(S)
    B = torch.diag_embed(S) @ Vt
    
    X = X.to('cpu')
    
    return A, B

def COTAN(W, X, r):
    X = X.to(W.device)
    
    matrix = W.float() @ X.float()
    matrix = matrix.float()
            
    U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    Vt = Vt.T
        
    A = U[:, :r]
    B = U[:, :r].T @ W
    
    T, S, Vt = torch.linalg.svd(B, full_matrices=False)
    S = torch.sqrt(S)
    A = A @ T @ torch.diag_embed(S)
    B = torch.diag_embed(S) @ Vt
    
    X = X.to('cpu')
    return A, B

def COTAN_BAD(W, X, r):
    X = X.to(W.device).float()
    C = X @ X.T
    eps = 1e-2
    I_eps = eps * torch.eye(C.shape[0]).to(C.device)
    L = torch.linalg.cholesky(C + I_eps)
    
    matrix = (W.float() @ L).float()            
    U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    Vt = Vt.T
        
    A = U[:, :r]
    B = torch.diag_embed(S[:r]) @ Vt[:r, :] @ torch.linalg.inv(L)
    
    X = X.to('cpu')
    return A, B

def COTAN_HALF(W, X, r):
    X = X.to(W.device).float()
    S, U = torch.linalg.eigh(X @ X.T)
    S[S < 0] = 0
    S = torch.sqrt(torch.sqrt(S))
    matrix = W.float() @ U @ torch.diag_embed(S)
    matrix = matrix.float()
    U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    Vt = Vt.T
        
    A = U[:, :r]
    B = U[:, :r].T @ W
    
    T, S, Vt = torch.linalg.svd(B, full_matrices=False)
    S = torch.sqrt(S)
    A = A @ T @ torch.diag_embed(S)
    B = torch.diag_embed(S) @ Vt
    
    X = X.to('cpu')
    return A, B

def SVF(W, r):
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    return U[:, -r:], S[-r:], Vt[-r:, :]

