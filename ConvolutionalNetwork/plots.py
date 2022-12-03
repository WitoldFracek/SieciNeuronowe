import matplotlib.pyplot as plt
import json
import numpy as np

with open('./experiments.json', 'r', encoding='utf-8') as file:
    DATA = json.load(file)


def pooling_results(metric: str, validation=False):
    prefix = 'val_' if validation else ''
    name = metric.capitalize().replace('_', ' ')
    metric = prefix + metric
    elements = len(DATA["Pool_MaxPooling2D_size_2"][metric]) + 1
    plt.plot(range(1, elements), DATA["Pool_MaxPooling2D_size_2"][metric], c='#3366ff',
             label='Max size 2')
    plt.plot(range(1, elements), DATA["Pool_MaxPooling2D_size_3"][metric], c='#3333cc',
             label='Max size 3')
    plt.plot(range(1, elements), DATA["Pool_MaxPooling2D_size_4"][metric], c='#666699',
             label='Max size 4')
    plt.plot(range(1, elements), DATA["Pool_MaxPooling2D_size_4"][metric], c='#66ccff',
             label='Max size 5')
    plt.plot(range(1, elements), DATA["Pool_AveragePooling2D_size_2"][metric], c='#ff9933',
             label='Avg size 2')
    plt.plot(range(1, elements), DATA["Pool_AveragePooling2D_size_3"][metric], c='#ff6600',
             label='Avg size 3')
    plt.plot(range(1, elements), DATA["Pool_AveragePooling2D_size_4"][metric], c='#cc0000',
             label='Avg size 4')
    plt.plot(range(1, elements), DATA["Pool_AveragePooling2D_size_4"][metric], c='#990000',
             label='Avg size 5')
    plt.legend()
    plt.title(f"{name} for different pooling types and sizes" + f"{' (validation)' if validation else ''}")
    plt.xlabel("epoch")
    plt.ylabel(name.lower())
    plt.show()


def augmentation_results(metric: str, validation=False):
    prefix = 'val_' if validation else ''
    metric_name = ' '.join(metric.capitalize().split('_'))
    metric = prefix + metric
    keys = [k for k in DATA.keys() if 'augmentation' in k]
    colors = ['#0099ff', '#66cc33', '#ff9900', '#cc0000']
    for key, color in zip(keys, colors):
        data = DATA[key][metric]
        label = ' and '.join(key.split('_')[1:])
        plt.plot(range(1, len(data) + 1), data, c=color, label=label)
        plt.legend()
    plt.title(f"{metric_name} for different types of augmentation" + f"{' (validation)' if validation else ''}")
    plt.xlabel('epoch')
    plt.ylabel(metric_name.lower())
    plt.show()


def confusion_matrix(keys, subplots, title, keys_names=None, validation=False, pad=2.5, keys_mapping=None, fontsize=10,
                     inchartfontsize=7, axis_turnoff=None):
    if keys_mapping is not None:
        keys_names = [keys_mapping(key) for key in keys]
    prefix = 'val_' if validation else ''
    labels = ['positive', 'negative']
    fig, axes = plt.subplots(subplots[0], subplots[1])
    fig.tight_layout(pad=pad)
    axes_flat = [ax for row in axes for ax in row]
    for ax, key, name in zip(axes_flat, keys, keys_names):
        arr = np.array([[
            DATA[key][f'{prefix}true_positives_1'][-1],
            DATA[key][f'{prefix}false_negatives_1'][-1]
        ], [
            DATA[key][f'{prefix}false_positives_1'][-1],
            DATA[key][f'{prefix}true_negatives_1'][-1]
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
    plt.suptitle(title + f"{' (validation)' if validation else ''}")
    plt.show()


def cnn_vs_mlp_results(metric, validation=False):
    prefix = 'val_' if validation else ''
    metric_name = ' '.join(metric.capitalize().split('_'))
    metric = prefix + metric
    keys = [key for key in DATA.keys() if 'mlp_vs_cnn' in key]
    colors = ['#0099ff', '#66cc33', '#ff9900', '#cc0000', '#9932CD']
    for key, color in zip(keys, colors):
        data = DATA[key][metric]
        label = key.split('_')[-1]
        plt.plot(range(1, len(data) + 1), data, c=color, label=label)
        plt.legend()
    plt.title(f"{metric_name} for different types of neural networks" + f"{' (validation)' if validation else ''}")
    plt.xlabel('epoch')
    plt.ylabel(metric_name.lower())
    plt.show()


def augmentation_name_map(name):
    xs = name.split('_')
    return ' and '.join(xs[1:])


def pooling():
    pooling_results('categorical_accuracy')
    pooling_results('categorical_accuracy', validation=True)
    pooling_results('loss')
    pooling_results('loss', validation=True)
    pooling_results("mean_squared_error")
    pooling_results("mean_squared_error", validation=True)
    confusion_matrix(sorted([k for k in DATA.keys() if 'Pool' in k]),
                     (2, 4), 'TP, FN, FP, TN for different pooling types and sizes', fontsize=7,
                     keys_mapping=lambda key: ' size '.join([key.split('_')[1], key.split('_')[-1]]))
    confusion_matrix(sorted([k for k in DATA.keys() if 'Pool' in k]), (2, 4),
                     'TP, FN, FP, TN for different pooling types and sizes', validation=True, fontsize=7,
                     keys_mapping=lambda key: ' size '.join([key.split('_')[1], key.split('_')[-1]]))


def augmentation():
    augmentation_results('loss')
    augmentation_results('loss', True)
    augmentation_results('categorical_accuracy')
    augmentation_results('categorical_accuracy', True)
    augmentation_results('mean_squared_error')
    augmentation_results('mean_squared_error', True)
    confusion_matrix([key for key in DATA.keys() if 'augmentation' in key], (2, 2),
                     'TP, FN, FP, TN for different augmentation methods', keys_mapping=augmentation_name_map,
                     fontsize=12, pad=4.5, inchartfontsize=10)
    confusion_matrix([key for key in DATA.keys() if 'augmentation' in key], (2, 2),
                     'TP, FN, FP, TN for different augmentation methods', keys_mapping=augmentation_name_map,
                     fontsize=12, pad=4.5, inchartfontsize=10, validation=True)


def augmentation_vs_cnn_results(metric, validation=False):
    prefix = 'val_' if validation else ''
    metric_name = ' '.join(metric.capitalize().split('_'))
    metric = prefix + metric
    keys = [key for key in DATA.keys() if 'cnn_vs_aug' in key]
    colors = ['#0099ff', '#ff9900', '#cc0000']
    for key, color in zip(keys, colors):
        data = DATA[key][metric]
        label = " ".join(key.split('_')[3:])
        plt.plot(range(1, len(data) + 1), data, c=color, label=label)
    plt.legend()
    plt.title(f"{metric_name} for different augmentation methods" + f"{' (validation)' if validation else ''}")
    plt.xlabel('epoch')
    plt.ylabel(metric_name.lower())
    plt.show()


def augmentation_vs_cnn():
    def name_mapping(key):
        xs = key.split('_')
        return ' '.join(xs[3:]).lower()
    augmentation_vs_cnn_results('mean_squared_error')
    augmentation_vs_cnn_results('mean_squared_error', True)
    augmentation_vs_cnn_results('categorical_accuracy')
    augmentation_vs_cnn_results('categorical_accuracy', True)
    confusion_matrix([key for key in DATA.keys() if 'cnn_vs_aug' in key], (2, 2),
                     "TP, FN, FP, TN for different augmentation methods", keys_mapping=name_mapping,
                     axis_turnoff=[(1, 1)], validation=False, fontsize=10, inchartfontsize=10, pad=4)
    confusion_matrix([key for key in DATA.keys() if 'cnn_vs_aug' in key], (2, 2),
                     "TP, FN, FP, TN for different augmentation methods", keys_mapping=name_mapping,
                     axis_turnoff=[(1, 1)], validation=True, fontsize=10, inchartfontsize=10, pad=4)

def cnn_vs_mlp():
    cnn_vs_mlp_results('categorical_accuracy')
    cnn_vs_mlp_results('categorical_accuracy', True)
    cnn_vs_mlp_results('mean_squared_error')
    cnn_vs_mlp_results('mean_squared_error', True)
    confusion_matrix(sorted([key for key in DATA.keys() if 'mlp_vs_cnn' in key]), (2, 3),
                     'TP, FN, FP, TN for different neural networks',
                     keys_mapping=lambda x: x.split('_')[-1],
                     fontsize=12, pad=4.5, inchartfontsize=8, axis_turnoff=[(1, 2)])
    confusion_matrix(sorted([key for key in DATA.keys() if 'mlp_vs_cnn' in key]), (2, 3),
                     'TP, FN, FP, TN for different neural networks',
                     keys_mapping=lambda x: x.split('_')[-1], validation=True,
                     fontsize=12, pad=4.5, inchartfontsize=8, axis_turnoff=[(1, 2)])


if __name__ == '__main__':
    pooling()
    # augmentation()
    # cnn_vs_mlp()
    augmentation_vs_cnn()


