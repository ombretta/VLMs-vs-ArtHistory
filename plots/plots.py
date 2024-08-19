import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def scatter_plot(x_variables, y_variables, labels, colors, figpath,
                 errorbar=False, title='', xlabel='', ylabel='',
                 xticklabels=None, xtickpos=None, xrotation=0,
                 yticklabels=None, ytickpos=None, yrotation=0,
                 xlim=None, ylim=None, ncols=1, nrows=1,
                 figsize=(5, 15), fontsize=20):
    """Makes a scatter plot with error bars for the distributions."""

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, (x, y) in enumerate(zip(x_variables, y_variables)):
        ax.scatter(x=x, y=y, label=labels[i], color=colors[i])
        if errorbar:
            ax.errorbar(x=x, y=np.mean(y), yerr=np.std(y), fmt='o', color="red")

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if xticklabels:
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xticklabels, rotation=xrotation, fontsize=fontsize)

    if yticklabels:
        ax.set_yticks(ytickpos)
        ax.set_yticklabels(yticklabels, rotation=yrotation, fontsize=fontsize)

    if ylim:
        ax.set_ylim(ylim)

    if xlim:
        ax.set_xlim(xlim)

    plt.grid(False)
    plt.savefig(figpath, bbox_inches='tight')


def plot_confusion_matrix(conf_matrix, figpath, figsize=(35, 20), labels=None, title=None, accuracy="",
                          cmap=None, xticks_rotation=90):
    """Plots a confusion matrix from the results of a classifier."""

    # Get the number of classes
    n_gt_classes = conf_matrix.shape[0]
    n_pred_classes = conf_matrix.shape[1]

    # Create figure and axis objects
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the confusion matrix
    if cmap == None:
        cmap = cm.get_cmap('plasma')
    im = ax.imshow(conf_matrix, cmap=cmap)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_pred_classes))
    ax.set_yticks(np.arange(n_gt_classes))

    print("labels", labels)

    if labels == None:
        gt_labels = np.arange(n_gt_classes)
        pred_labels = np.arange(n_pred_classes)

    elif type(labels) is tuple:
        gt_labels = labels[0]
        pred_labels = labels[1]

    else:
        gt_labels = labels
        pred_labels = labels

    print("n_gt_classes", len(np.arange(n_gt_classes)))
    print("n_pred_classes", len(np.arange(n_pred_classes)))
    print(len(gt_labels), gt_labels)
    print(len(pred_labels), pred_labels)

    ax.set_xticklabels(pred_labels, rotation=xticks_rotation, fontsize=20, horizontalalignment='right')
    ax.set_yticklabels(gt_labels, fontsize=20)

    # Rotate the x-axis labels if needed
    plt.xticks()

    # Loop over data dimensions and create text annotations
    for i in range(n_gt_classes):
        for j in range(n_pred_classes):
            text = ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="w", fontsize=15)

    # Set labels and title
    ax.set_xlabel('Predicted class', fontsize=30)
    ax.set_ylabel('True class', fontsize=30)

    if title is None:
        title = 'Confusion Matrix'+accuracy

    ax.set_title(title, fontsize=30)

    # Show color bar
    # plt.colorbar(im)

    # Save the plot
    plt.savefig(figpath, bbox_inches='tight')
    plt.close()

