import numpy as np
from lib import Globals
import torch
import tqdm

def PISSA(W, r):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    S_root = torch.sqrt(S_r)
        
    U_mod = U_r @ torch.diag_embed(S_root)
    V_mod = torch.diag_embed(S_root) @ Vh_r
    
    return U_mod, V_mod

def CORDA_MY(W, X, r):
    X = X.to('cuda')
    C = X @ X.T
    C = C.float()
    
    matrix = W @ C
    if not torch.isfinite(matrix).all():
        print("Матрица содержит не-конечные значения на следующих индексах:")

    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        
    # U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    # Vt = Vt.T
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    S = torch.sqrt(S)
        
    A = U[:, :r] @ torch.diag_embed(S[:r])
    
    # B = torch.diag_embed(S[:r]) @ (Vt[:r, :] @ torch.linalg.inv(C))
    B = torch.diag_embed(1 / S[:r]) @ (U.T[:r, :] @ W)
    X = X.to('cpu')
    return A, B

def CORDA(W, X, r):
    X = X.to('cuda')
    C = X @ X.T
    C = C.float()
    
    matrix = W @ C
    if not torch.isfinite(matrix).all():
        print("Матрица содержит не-конечные значения на следующих индексах:")

    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        
    U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    Vt = Vt.T
    # U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    # S = torch.sqrt(S)
        
    A = U[:, :r] @ torch.diag_embed(S[:r])
    
    # C_inv = torch.linalg.inv(C)
    I = torch.eye(C.size(0)).to('cuda')

    C_inv = torch.linalg.lstsq(I, C).solution
    print(C_inv.shape)

    B = torch.diag_embed(S[:r]) @ (Vt[:r, :] @ C_inv)
    # B = torch.diag_embed(1 / S[:r]) @ (U.T[:r, :] @ W)
    X = X.to('cpu')
    return A, B

def SCORDA(W, X, r):
    X = X.to('cuda')
    print('W, X, r =', W.shape, X.shape, r)
    print('Pref', W.dtype, X.dtype)
    
    matrix = W.float() @ X.float()
    matrix = matrix.float()
    if not torch.isfinite(matrix).all():
        print("Матрица содержит не-конечные значения на следующих индексах:")

    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        
    # U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    U, S, Vt = torch.svd_lowrank(matrix, q=r + 8, niter=10)
    Vt = Vt.T
        
    A = U[:, :r]
    
    # B = torch.diag_embed(S[:r]) @ (Vt[:r, :] @ torch.linalg.inv(C))
    B = U[:, :r].T @ W
    # if X.shape[0] != X.shape[1]:
    #     T, _, _ = torch.linalg.svd(X.float(), full_matrices=False)
    #     prev = torch.norm(B)
    #     B = (U[:, :r].T @ W @ T) @ T.T
    #     print(prev - torch.norm(B), "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    # print('ready inv')
    X = X.to('cpu')
    # print(W.dtype, X.dtype, C.dtype, A.dtype, B.dtype)
    # print("AB shape", A.shape, B.shape)
    return A, B

def SVF(W, r):
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    return U[:, -r:], S[-r:], Vt[-r:, :]

def JustSVD(W, X, r):
    return Globals.compress_matrix_full(W, r)

def MyFindWeightNorm(W, X, r):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    A_r, _, _ = Globals.compress_matrix(W @ U @ np.diag(S), r)
    return A_r @ A_r.T @ W @ U @ U.T

def SuperFindWeightNorm(W, X, r):
    Q, R = np.linalg.qr(X)
    A_r, _, _ = Globals.compress_matrix(W @ X, r)
    return A_r @ A_r.T @ W @ Q @ Q.T


def MyFindWeightNormOLD(W, X, r):
    U, Sigma, VT = np.linalg.svd(X)
    _, Sigma, _ = np.linalg.svd(X, full_matrices=False)
    n, m = X.shape
    pad = np.full(n - m, 1)
    Sigma = np.concatenate((Sigma, pad))
    S = U @ np.diag(Sigma)
    T = W @ S
    T1 = T[:, :m]
    T1_ = Globals.compress_matrix_full(T1, r)
    zero_columns = np.zeros((T1_.shape[0], n - m))
    T1_ = np.hstack((T1_, zero_columns))
    return T1_ @ np.diag(1 / Sigma) @ U.T

def Whitening(X, eps):
    X_mod = X @ X.T
    X_mod += np.eye(X_mod.shape[0]) * 1e-2
    return np.linalg.cholesky(X_mod)


def AuthorFindWeightNorm(W, X, r):
    S = Whitening(X, 1e-2)
    return Globals.compress_matrix_full(W @ S, r) @ np.linalg.inv(S)



def CompressSpecial(position, X, mask, lays, compressor, inputs):
    handles = [{} for _ in range(lays)]
    W_new = [{} for _ in range(lays)]

    def calculate(lay, name, W, r):
        def hook(model, input):
            X = input[0][0].clone().cpu().detach().numpy().T
            # W_ = compressor(W.copy(), X.copy(), r)
            # W_new[lay][name] = W
            # print(np.linalg.norm(W - W_))
            # print(np.linalg.norm(W @ X - W_ @ X))
            print(W.shape)
            # print(X)
            aa=5

        return hook

    for i in range(lays):
        for name, r in position.names.items():
            W = position.get(i, name).weight.clone().cpu().detach().numpy().T
            handles[i][name] = position.get(i, name).register_forward_pre_hook(calculate(i, name, W, r))

    # with torch.no_grad():
    #     position.model(X, attention_mask=mask)
    
    # for param in position.model.parameters():
    #     param.requires_grad = False
    
    # TRAINER.train()
    
    # TRAINER.compute_loss(position.model, inputs)

    for i in range(lays):
        for name in handles[i].keys():
            handles[i][name].remove()
            

    for i in range(lays):
        for name, r in position.names.items():
            with torch.no_grad():
                kekke = 4
                # W = W_new[i][name]
                # KEK = position.get(i, name).weight.cpu()
                # print(np.linalg.norm(KEK - W.T))
                # position.get(i, name).weight.copy_(torch.from_numpy(W.T))


def CompressSVD(position, lays):
    for i in tqdm.trange(lays):
        for name, r in position.names.items():
            W = position.get(i, name).weight.clone().cpu().detach().numpy().T
            W_new = Globals.compress_matrix_full(W, r)
            with torch.no_grad():
                position.get(i, name).weight.copy_(torch.from_numpy(W_new.T))



