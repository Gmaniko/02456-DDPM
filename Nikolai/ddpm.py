import torch
from torch import nn, Tensor
import copy

from model import ScoreNetwork0


from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def preprocess(x):
    return 2*(ToTensor()(x))-1



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 1000
beta = torch.linspace(1e-4, 0.02, T, device=device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)



def calc_loss(model: nn.Module, loss_fn: nn.MSELoss, x: Tensor) -> Tensor:
    t = torch.randint(0, T, size=(x.shape[0],1), device=device)
    t = t[..., None, None].expand(*t.shape, *x.shape[-2:])
    ab_t = alpha_bar[t]
    eps = torch.randn_like(x, device=device)
    s = torch.sqrt(ab_t)*x + torch.sqrt(1-ab_t)*eps
    out = model(s, t)
    return loss_fn(eps, out)


def sampling(model: nn.Module):
    data_dim = model.dims
    x = torch.randn(size=data_dim, device=device)
    for t in range(T-1, -1, -1):
        z = torch.randn(size=data_dim) if t > 0 else torch.zeros(size=data_dim)
        z = z.to(device)
        t_tensor = t*torch.ones_like(x, device=device)
        out = model(x, t_tensor)
        sd = torch.sqrt(beta[t])
        x = 1/torch.sqrt(alpha[t])*(x - beta[t]/torch.sqrt(1-alpha_bar[t])*out) + sd*z
    return x

def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.MSELoss, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        # Compute prediction and loss
        loss = calc_loss(model, loss_fn, X.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.MSELoss):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    # test_loss, correct = 0, 0
    test_loss = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, _ in dataloader:
            L = calc_loss(model, loss_fn, X.to(device))
            test_loss += L.item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    print(f"Test Loss: {test_loss:>8f} \n")

    return test_loss





def train_ddpm(dataset, train_loader, test_loader, epochs, lr, patience=10):
    print("device: ", device)
    model = ScoreNetwork0(dataset).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = torch.inf 
    best_model_weights = None



    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loss = test_loop(test_loader, model, loss_fn)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            patience = 10  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_model_weights)

    print("Done!")

    return model