import torch
import matplotlib.pyplot as plt
import numpy as np
#Plot a single image
@torch.no_grad()
def plot_image(img, ax):
    #print(images.shape)
    cmap = 'gray'
    img = torch.squeeze(img)
    img = (img+1)/2 
    img = torch.clamp(img, 0.0, 1.0)
    if len(img.shape) == 3:
        cmap = None
        img = img.permute(1,2,0)
    ax.imshow(img.detach().cpu().numpy(), cmap=cmap)
    ax.set_axis_off()
    return ax

#Plot grid of images.
@torch.no_grad()
def grid_plot(images, grid_dim, fig_size):
    rows, cols = grid_dim
    if len(images) != rows*cols:
      raise ValueError("Number of images does not match grid dimensions.")
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=fig_size)
    for r in range(rows):
        for c in range(cols):
            plot_image(images[r+c*cols], axs[r,c])
    fig.tight_layout()
    return fig, axs