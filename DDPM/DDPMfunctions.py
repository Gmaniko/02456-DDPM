import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as func
#Forward process of the DDPM model
def forward_process(img,eps,t,ap1,ap2):
    #Add noise by using the closed formula derived in DDPM paper
    return ap1[t].view(-1,1,1,1)*img + ap2[t].view(-1,1,1,1)*eps
#Predict the noise in the picture by using a Unet model
def prediction(model,img,t):
    return model(img,t)
#Get_Loss is used in the training process. We generate random noise, add it to some timestep (t) for a given batch of pictures. 
#Then we predict the noise by using our model and return the simple loss function which is also just the MSE between the real noise and predicted noise.'
#We then take a gradient step with our optimizer in the training process 
def get_loss(model, batch, ts, device,ap1,ap2):
    eps = torch.randn(batch.shape).to(device)
    xt = forward_process(batch, eps,ts,ap1,ap2)
    eps_pred = prediction(model, xt, ts)
    return func.mse_loss(eps_pred,eps)
#The backward process is used for sampling pictures.
# A timestep, is given together with predicted noise for a sample x and then we remove this predicted noise from the picture
#This is done for T to 0 iterativly such that we end up with a completly denoised picture
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



