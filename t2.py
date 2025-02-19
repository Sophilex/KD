import torch
x = torch.tensor([
    [0, 0.0000001, 0.00000001, 0.0000001, 0.0000001],
    [5, 10, 15, 20, 25],
    [30, 25, 20, 15, 10]
], dtype=torch.float32)
y = torch.multinomial(x, 3, replacement=False)
print(y)
# x = torch.tensor([
#     [10, 20, 15, 30, 25],
#     [5, 10, 15, 20, 25],
#     [30, 25, 20, 15, 10]
# ])

# y = torch.tensor([
#     [0, 1],
#     [0, 1],
#     [1, 2]
# ])
# ans = torch.cat([x,y], dim=1)
# print(ans)
