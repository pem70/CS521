import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 0 # target class

epsReal = 0.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad
adv_x = x - eps*torch.sign(x.grad)

new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)

# repeat the experiment with different target class
t_1 = 1
epsReal_1 = 0.5
eps1 = epsReal_1 - 1e-7

x1 = x.detach().clone().requires_grad_(True)

# IMPORTANT: zero any stale grads and use the same loss L
N.zero_grad()
loss1 = L(N(x1), torch.tensor([t_1], dtype=torch.long))
loss1.backward()

# Targeted FGSM: move *against* the gradient for the target class
adv_x_1 = torch.clamp(x1 - eps1 * torch.sign(x1.grad), 0.0, 1.0)

new_class_1 = N(adv_x_1).argmax(dim=1).item()
print("New Class 1:", new_class_1)
assert new_class_1 == t_1

# Check closeness under L-infinity norm
linf_dist_1 = torch.norm((x1 - adv_x_1), p=float('inf')).item()
print("L_inf distance (t=1):", linf_dist_1)
assert linf_dist_1 <= epsReal_1 + 1e-8

t_1 = 1
eps = 1                
step = 0.05             
k = 20                  

x0 = x.detach()

x_adv_1 = x0 + torch.empty_like(x0).uniform_(-eps, eps)
x_adv_1 = torch.clamp(x_adv_1, 0.0, 1.0)

for _ in range(k):
    x_adv_1.requires_grad_(True)
    N.zero_grad()

    logits = N(x_adv_1)              
    target_logit = logits[:, t_1]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, t_1] = False
    top_other, _ = logits[mask].view(logits.size(0), -1).max(dim=1)
    loss = -(target_logit - top_other)

    loss.backward()

    with torch.no_grad():
        grad = x_adv_1.grad
        x_adv_1 = x_adv_1 - step * torch.sign(grad)
        x_adv_1 = torch.max(torch.min(x_adv_1, x0 + eps), x0 - eps)
        x_adv_1 = torch.clamp(x_adv_1, 0.0, 1.0)

    x_adv_1 = x_adv_1.detach()

    if N(x_adv_1).argmax(dim=1).item() == t_1:
        break

new_class_1 = N(x_adv_1).argmax(dim=1).item()
print("New Class 1 (targeted CW-PGD):", new_class_1)
linf_dist = torch.norm((x0 - x_adv_1), p=float('inf')).item()
print("L_inf distance:", linf_dist)
assert linf_dist <= eps + 1e-8
