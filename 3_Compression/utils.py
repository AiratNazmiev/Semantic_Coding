import numpy as np
import matplotlib.pyplot as plt


def plot_dataset(mnist_train):
    fig, axes = plt.subplots(ncols=8, nrows=3, figsize=(8, 4))
    axes = axes.flatten()

    for i, ax in enumerate(axes): 
        ax.imshow(mnist_train.data[i], cmap='gray')
        ax.set_title(f"Number {mnist_train.targets[i]}", fontsize=8)
        ax.set_axis_off()
        
    fig.tight_layout()
    
    
def plot_augmentation(x, transform):
    fig, axes = plt.subplots(ncols=8, nrows=3, figsize=(8, 4))
    fig.suptitle('Data augmentation')
    axes = axes.flatten()

    for i, ax in enumerate(axes): 
        if i == 0:
            ax.imshow(x, cmap='gray')
            ax.set_title(f"Original", fontsize=8)
        else:
            ax.imshow(transform(x.unsqueeze(0))[0], cmap='gray')
        ax.set_axis_off()
            
    fig.tight_layout()
    

def plot_learning_curve(loss_logs, num_epochs, semilogy=False):
    plt.figure()
    epochs = np.arange(1, num_epochs+1)
    if semilogy:
        plt.semilogy(epochs, loss_logs['train'])
        plt.semilogy(epochs, loss_logs['test'])
    else:
        plt.plot(epochs, loss_logs['train'])
        plt.plot(epochs, loss_logs['test'])
    plt.grid(True, 'major'); plt.grid(True, 'minor', lw=0.5, alpha=0.5)
    plt.xlim([1, num_epochs])
    plt.show()
    

def plot_data_latent_decoded(autoencoder, x, labels, device, lossless=True, lossy_rehsape=(8, 4)):
    if lossless:
        x_test_latent = autoencoder.encode(x[:24].unsqueeze(1).to(device), flatten=False).detach().squeeze().cpu().numpy().reshape(-1, 28, 28)
    else:
        x_test_latent = autoencoder.encode(x[:24].unsqueeze(1).to(device), flatten=False).detach().squeeze().cpu().numpy().reshape(24, *lossy_rehsape)
         
    x_test_reconstructed = autoencoder(x[:24].unsqueeze(1).to(device)).detach().squeeze().cpu().numpy()

    fig, axes = plt.subplots(ncols=8, nrows=9, figsize=(8, 9))
    axes = axes.flatten()

    img_idx = 0
    lat_idx = 0
    rec_idx = 0
    for i, ax in enumerate(axes): 
        if (i // 8) % 3 == 0:
            ax.set_title(f"[{img_idx}] Number {labels[img_idx]}:", fontsize=8)
            ax.imshow(x[img_idx], cmap='gray')
            img_idx += 1
        elif (i // 8) % 3 == 1:
            ax.set_title(f"[{lat_idx}] Latent:", fontsize=8)
            ax.imshow(x_test_latent[lat_idx], cmap='gray')
            lat_idx += 1
        else:
            ax.set_title(f"[{rec_idx}] Decoded:", fontsize=8)
            ax.imshow(x_test_reconstructed[rec_idx], cmap='gray')
            rec_idx += 1
            
        ax.set_axis_off()
        
    fig.tight_layout()
    
    
def plot_data_decoded_svd(x, x_rec, labels, rank):
    fig, ax = plt.subplots(ncols=8, nrows=6, figsize=(8, 8))
    fig.suptitle(f'SVD compression: rank={rank}')
    ax = ax.flatten()
    img_idx = 0
    rec_idx = 0
    for i in range(len(ax)):
        if (i // 8) % 2 == 0:
            ax[i].set_title(f"[{img_idx}] Number {labels[img_idx]}:", fontsize=8)
            ax[i].imshow(x[img_idx].cpu(), cmap='gray')
            img_idx += 1
        else:
            ax[i].set_title(f"[{rec_idx}] Decoded:", fontsize=8)
            ax[i].imshow(x_rec[rec_idx].cpu(), cmap='gray')
            rec_idx += 1
            
        ax[i].set_axis_off()
            
    fig.tight_layout()