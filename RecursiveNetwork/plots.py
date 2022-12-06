import matplotlib.pyplot as plt
import json
import numpy as np


with open('./experiments.json', 'r', encoding='utf-8') as file:
    DATA = json.load(file)


def confusion_matrix(keys, subplots, title, keys_names=None, validation=False, pad=2.5, keys_mapping=None, fontsize=10,
                     inchartfontsize=7, axis_turnoff=None, titlefontsize=15):
    if keys_mapping is not None:
        keys_names = [keys_mapping(key) for key in keys]
    prefix = 'val_' if validation else ''
    labels = ['positive', 'negative']
    fig, axes = plt.subplots(subplots[0], subplots[1])
    fig.tight_layout(pad=pad)
    fig.set_size_inches(5 * subplots[1], 5 * subplots[0])
    axes_flat = [ax for row in axes for ax in row]
    for ax, key, name in zip(axes_flat, keys, keys_names):
        arr = np.array([[
            DATA[key][f'{prefix}true_positives'][-1],
            DATA[key][f'{prefix}false_negatives'][-1]
        ], [
            DATA[key][f'{prefix}false_positives'][-1],
            DATA[key][f'{prefix}true_negatives'][-1]
        ]])
        ax.set_title(name, fontdict={'fontsize': fontsize})
        ax.text(0, 0, int(arr[0, 0]), ha="center", va="center", color="w", fontdict={'fontsize': inchartfontsize})
        ax.text(0, 1, int(arr[0, 1]), ha="center", va="center", color="w", fontdict={'fontsize': inchartfontsize})
        ax.text(1, 0, int(arr[1, 0]), ha="center", va="center", color="w", fontdict={'fontsize': inchartfontsize})
        ax.text(1, 1, int(arr[1, 1]), ha="center", va="center", color="w", fontdict={'fontsize': inchartfontsize})
        ax.imshow(arr, cmap='Spectral')
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(2))
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    if axis_turnoff is not None:
        for x, y in axis_turnoff:
            axes[x][y].set_visible(False)
    plt.suptitle(title + f"{' (validation)' if validation else ''}", fontsize=titlefontsize)
    plt.show()


def embedding_names_mapping(key):
    xs = key.split('_')
    return f"Space: {xs[1]}\nType: {xs[2]}"


def masking_names_mapping(key):
    xs = key.split('_')
    return f"Enabled: {xs[1]}\nType: {xs[2]}"


def padding_names_mapping(key):
    xs = key.split('_')
    return f"Padding size: {xs[1]}\nType: {xs[2]}"


def embedding_results(metric: str, validation=False):
    prefix = 'val_' if validation else ''
    metric_name = ' '.join(metric.capitalize().split('_'))
    metric = prefix + metric
    keys = sorted([k for k in DATA.keys() if "Embedding" in k], key=lambda x: x.split('_')[2])
    colors = ['#cc3300', '#ff6600', '#660000', '#3399ff', '#003399', '#0099cc']
    for key, color in zip(keys, colors):
        data = DATA[key][metric]
        label = ' and '.join(key.split('_')[1:])
        plt.plot(range(1, len(data) + 1), data, c=color, label=label)
        plt.legend()
    plt.title(f"{metric_name} for different embedding size" + f"{' (validation)' if validation else ''}")
    plt.xlabel('epoch')
    plt.ylabel(metric_name.lower())
    plt.show()


def embedding_test():
    embedding_results('mean_squared_error', False)
    embedding_results('binary_accuracy', False)
    embedding_results('mean_squared_error', True)
    embedding_results('binary_accuracy', True)
    confusion_matrix(
        sorted([k for k in DATA.keys() if "Embedding" in k], key=lambda x: x.split('_')[2]),
        (2, 3), "TN, TP, FN, FP for different sizes of embedding",
        keys_mapping=embedding_names_mapping, inchartfontsize=20, fontsize=15, titlefontsize=20)


def masking_test():
    masking_results('mean_squared_error', False)
    masking_results('binary_accuracy', False)
    masking_results('mean_squared_error', True)
    masking_results('binary_accuracy', True)
    confusion_matrix(
        sorted([k for k in DATA.keys() if "Masking" in k], key=lambda x: x.split('_')[2]),
        (2, 2), "TN, TP, FN, FP for masking on and off",
        keys_mapping=masking_names_mapping, inchartfontsize=20, fontsize=15, titlefontsize=20, pad=3.5)


def masking_results(metric: str, validation=False):
    prefix = 'val_' if validation else ''
    metric_name = ' '.join(metric.capitalize().split('_'))
    metric = prefix + metric
    keys = sorted([k for k in DATA.keys() if "Masking" in k], key=lambda x: x.split('_')[2])
    colors = ['#cc3300', '#ff6600', '#3399ff', '#003399']
    for key, color in zip(keys, colors):
        data = DATA[key][metric]
        xs = key.split('_')
        label = f'{xs[2]}: {xs[1]}'
        plt.plot(range(1, len(data) + 1), data, c=color, label=label)
        plt.legend()
    plt.title(f"{metric_name} for masking states" + f"{' (validation)' if validation else ''}")
    plt.xlabel('epoch')
    plt.ylabel(metric_name.lower())
    plt.show()


def padding_test():
    padding_results('mean_squared_error', False)
    padding_results('binary_accuracy', False)
    padding_results('mean_squared_error', True)
    padding_results('binary_accuracy', True)
    confusion_matrix(
        sorted([k for k in DATA.keys() if "Padding" in k], key=lambda x: x.split('_')[2]),
        (2, 3), "TN, TP, FN, FP for different sizes of padding",
        keys_mapping=padding_names_mapping, inchartfontsize=20, fontsize=15, titlefontsize=20)


def padding_results(metric: str, validation=False):
    prefix = 'val_' if validation else ''
    metric_name = ' '.join(metric.capitalize().split('_'))
    metric = prefix + metric
    keys = sorted([k for k in DATA.keys() if "Padding" in k], key=lambda x: x.split('_')[2])
    colors = ['#cc3300', '#ff6600', '#660000', '#3399ff', '#003399', '#0099cc']
    for key, color in zip(keys, colors):
        data = DATA[key][metric]
        xs = key.split('_')
        label = f'{xs[2]}: {xs[1]}'
        plt.plot(range(1, len(data) + 1), data, c=color, label=label)
        plt.legend()
    plt.title(f"{metric_name} for different padding size" + f"{' (validation)' if validation else ''}")
    plt.xlabel('epoch')
    plt.ylabel(metric_name.lower())
    plt.show()


if __name__ == '__main__':
    # embedding_test()
    # masking_test()
    # padding_test()



