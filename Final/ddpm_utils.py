import os
import torch
from torch.nn.functional import mse_loss
from torchvision.utils import save_image 
from tqdm import tqdm

#Inspired from ...
def _cosine_schedule(T, s=0.008):
    y = torch.linspace(0,T, T+1)
    alphabar = torch.cos(((y / T)+s) / (1+s)*torch.pi*0.5)**2
    alphabar = alphabar / alphabar[0]
    beta = 1 - (alphabar[1:] / alphabar[:-1])
    return torch.clip(beta, 1e-4, 0.999)
    

def variance_schedule(T:int=1000, schedule:str='linear', device:torch.device|str='cpu'):
    match schedule:
        case 'linear':
            beta = torch.linspace(1e-4, 0.02, T)
        case 'quadratic':
            beta = torch.linspace(1e-4 ** 0.5, 0.02 ** 0.5, T) ** 2
        case 'cosine':
            beta = _cosine_schedule(T)
        case _:
            raise ValueError("Valid schedules are 'linear', 'quadratic' and 'cosine'.")
    
    beta = beta.to(device)
    alpha = 1 - beta 
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

# Forward process of the DDPM model
def forward_process(x0: torch.Tensor, t, alpha_bar):
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    
    t = t.view(-1,1,1,1).to(x0.device)

    abar_t = alpha_bar[t]
    eps = torch.randn_like(x0, device=x0.device)
    #Add noise by using the closed formula derived in DDPM paper
    x_t = abar_t**0.5 * x0 + (1-abar_t)**0.5 * eps
    return x_t, eps


# The backward process is used for sampling pictures.
# A timestep, is given together with predicted noise for a sample x and then we remove this predicted noise from the picture
# This is done for T to 0 iterativly such that we end up with a completly denoised picture
def backward_process(eps_pred,t,x,alpha,beta,ap,device):
    #Variance which is included to make sure the DDPM model is not deterministic in the sampling approach.
    var = torch.zeros_like(eps_pred).to(device)
    c1 = 1/alpha**0.5
    c2 = (1-alpha)/ap
    #The mean of the new picture (reparimzation trick)
    ps = c1[t].view(-1,1,1,1)*(x-c2[t].view(-1,1,1,1)*eps_pred)
    #if t > 0 we want to include variance in the predicted normal distribuation
    #if t = 0 we want a deterministic denoised picture hence we do not include variance there
    if t > 0:
        noise = torch.randn_like(eps_pred).to(device)
        #Beta tilde (posterior variance) can be tested too
        var = (beta[t]**0.5)*noise
    ps = ps + var
    return ps


# compute_loss is used in the training process. We generate random noise, add it to some timestep (t) for a given batch of pictures. 
# Then we predict the noise by using our model and return the simple loss function which is also just the MSE between the real noise and predicted noise.'
# We then take a gradient step with our optimizer in the training process 
def _compute_loss(model, X, alpha_bar, T:int=1000):
    # Sample t uniformly
    t = torch.randint(low=0, high=T, size=(X.shape[0],), device=X.device)
    # Add noise in the forward process
    X_t, eps = forward_process(X, t, alpha_bar)
    # Predict noise
    pred = model(X_t, t)
    # Compute MSE
    return mse_loss(pred, eps)


def train_loop(dataloader, model, alpha_bar, optimizer, n_epochs:int, device):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    T = len(alpha_bar)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, _) in enumerate(dataloader):
            X = X.to(device)

            # Reset gradient and compute loss
            optimizer.zero_grad()
            loss = _compute_loss(model, X, alpha_bar, T)

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
            # if batch % 5 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Done!")
    return 

# Samples 'batch_size' images 
@torch.no_grad()
def sampling(model, var_param, device, dataset_name:str='mnist', batch_size:int=1):
    
    if dataset_name == 'mnist':
        img_dim = (1,28,28)
    elif dataset_name == 'cifar10':
        img_dim = (3,32,32)
    else:
        raise ValueError("Invalid dataset name. Either 'mnist' or 'cifar10'.")
    
    beta, alpha, alpha_bar = var_param
    T = len(beta)
    model.eval()
    X_T = torch.randn(size=(batch_size, *img_dim), device=device)
    X_t = X_T

    for t in reversed(range(T-20)):
        tt = torch.full(size=(batch_size,), fill_value=t, dtype=torch.int, device=device)
        eps_pred = model(X_t, tt)

        X_t = reverse_process(eps_pred, t, X_t, beta, alpha, alpha_bar)

    X_0 = X_t
    return X_0
        

# The backward process is used for sampling pictures.
# A timestep, is given together with predicted noise for a sample x and then we remove this predicted noise from the picture
# This is done for T to 0 iterativly such that we end up with a completly denoised picture
def reverse_process(eps_pred, t, X_t, beta, alpha, alpha_bar):
    c1 = 1/alpha[t]**0.5
    c2 = (1-alpha[t]) / (1-alpha_bar[t])**0.5
    # The mean of the new picture (reparimzation trick)
    X_t = c1 * (X_t - c2*eps_pred)
    # if t > 0, we include variance in the predicted normal distribution
    # if t = 0, we want a deterministic denoised picture hence we do not include variance
    if t > 0:
        z = torch.randn_like(X_t, device=X_t.device)
        # Beta tilde (posterior variance) can be tested too
        X_t += beta[t]**0.5 * z 
    return X_t


@torch.no_grad()
def generate_samples_todir(model, dataset_name, output_dir, var_params, num_samples, batch_size, device):
    for i in tqdm(range(0, num_samples, batch_size)):
        x0 = sampling(model, var_params, device,
                      dataset_name, batch_size)
        x0 = ((x0 + 1) / 2)
        for j in range(x0.size(0)):
            if j == 0:
                print(f"{i}")
            save_image(x0[j], os.path.join(output_dir, f"{i+j}.png"))