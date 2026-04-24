import torch

# X, y, beta are tensors on the GPU
def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):


    max_iter = 20
    reg = 1e-6  

    for _ in range(max_iter):
        z = X @ beta
        p = torch.sigmoid(z)
        p = torch.clamp(p, min=1e-8, max=1-1e-8)

        grad = X.T @ (p - y) + reg * beta

        W = p * (1 - p)
        XW = X * W.unsqueeze(1)
        H = X.T @ XW + reg * torch.eye(n_features, device=X.device)

        delta = torch.linalg.solve(H, grad)
        beta[:] = beta - delta