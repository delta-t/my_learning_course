import torch

alpha = 0.001
w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
optimizer = torch.optim.SGD([w], lr=alpha)
for _ in range(500):
    function = torch.prod(torch.log(torch.log(w + 7)))
    function.backward()
    optimizer.step()
    optimizer.zero_grad()

print(w)
