import torch

if __name__ == "__main__":
    # x = torch.ones(size=(3, 4, 5))
    x = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],

        [[9, 8, 7],
         [6, 5, 4],
         [3, 2, 1]]
    ])
    print(x.shape)
    y = torch.sum(x, dim=-1, keepdim=True)
    print(y)
    print(x/y)