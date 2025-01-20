import sounddevice as sd
import time
import matplotlib.pyplot as plt


def play_audios(a, Fs, norm_coef=1., pause=1.):
    for i in range(a.shape[0]):
        sd.play(a[i, :]*norm_coef, samplerate=Fs)
        time.sleep(len(a[i, :])/Fs + pause)
        
        
def plot_matrix(mat, title):
    plt.figure(figsize=(5, 3), dpi=200)
    plt.title(title)
    plt.imshow(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f'{mat[i, j]:.3f}', ha="center", va="center", color="k")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()