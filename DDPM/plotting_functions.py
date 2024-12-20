import torch
import matplotlib as plt
import numpy as np
#Plot a single image either directly or as a part of a grid
@torch.no_grad()
def plot_image(images, grid=False):
    #print(images.shape)
    if len(images.shape) == 4:
        images = torch.squeeze(images)
    img = torch.clamp(images,-1.0,1.0)
    img = (img+1)/2     
    if len(images.shape == 3):
        plt.imshow(img.detach().cpu().permute(1,2,0).numpy())
        plt.axis('off')
    else:
        plt.imshow(img.detach().cpu().numpy())
        plt.axis('off')
    if grid:
        pass
    else:
        plt.show()

#Plot grid of images.
@torch.no_grad()
def grid_plot(images):
    figs = plt.figure(figsize=(10, 10))
    row = int(len(images) ** (1 / 2))
    col = round(len(images) / row)
    index = 0
    for r in range(row):
        for c in range(col):
            figs.add_subplot(row, col, index + 1)
            if index < len(images):
                plot_image(images[index], grid=True)
                index += 1
    
    plt.show()