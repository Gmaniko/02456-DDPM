
import torch
#Inspired from ...
def cosine_schedule(T, s=0.008):
    y = torch.linspace(0,T, T+1)
    alphabar = torch.cos(((y / T)+s) / (1+s)*torch.pi*0.5)**2
    alphabar = alphabar / alphabar[0]
    beta = 1 - (alphabar[1:] / alphabar[:-1])
    return torch.clip(beta, 0.0001, 0.999)

def varianceschedule(device,T=1000,schedule='linear'):
    if schedule == 'linear':
        beta = torch.linspace(0.0001,0.02,T).to(device)
        return beta
    elif schedule == 'quadratic':
        beta = (torch.linspace(0.0001 ** 0.5 ,0.02 ** 0.5,T)**2).to(device)
        return beta
    elif schedule == 'cosine':
        beta = cosine_schedule(1000)
        beta = beta.to(device)
        return beta