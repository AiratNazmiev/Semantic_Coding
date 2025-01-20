import numpy as np
import matplotlib.pyplot as plt


def hartley_entropy(dictionary):
    N = len(dictionary)
    return np.log2(N)


def entropy(dictionary):
    freqs = np.array(list(dictionary.values()))
    total_freq = np.sum(freqs)
    probs = freqs / total_freq
    
    H = np.sum(-probs * np.log2(probs))
    
    return H


def word_avg_len(dictionary):
    total_num = 0
    avg_len = 0.
    for w, n in dictionary.items():
        total_num += n
        avg_len += n * len(w)
        
    avg_len /= total_num
    
    return avg_len


def plot_barchart(sorted_tuples, plot_first=None, log=False, file=None):
    plt.figure(figsize=(10, 3), dpi=200)
    if plot_first is None:
        plot_first = len(sorted_tuples)
    bar_labels = [x[0] for x in sorted_tuples[:plot_first]]
    bar_freqs = [x[1] for x in sorted_tuples[:plot_first]]
    plt.bar(bar_labels, bar_freqs)
    if log:
        plt.yscale('log')
      
    plt.grid(True, 'major', lw=0.5, alpha=0.5)  
    plt.grid(True, 'minor', alpha=0.1, lw=0.5)
    
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()