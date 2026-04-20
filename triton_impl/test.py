import torch
if __name__ == "__main__":
    S = torch.tensor([[0.5, 2.0, 3.0]])
    print(S.stride())