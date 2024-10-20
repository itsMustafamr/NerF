import torch

def simulator(x, t=3):
    
    x0, v0 = x[:2], x[2:]

    a = torch.tensor([0, -9.81])
    v = v0 + a * t
    x = x0 + v0 * t + .5 * a * t**2
        
    return torch.cat((x, v))