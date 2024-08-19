import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score

style_labels = {
    "WikiArt": ['Abstract Expression.', 'Action Painting', 'Analytical Cubism', 'Art Nouveau Modern', 'Baroque',
                'Color Field Painting', 'Contemporary Realism', 'Cubism', 'Early Renaissance', 'Expressionism',
                'Fauvism', 'High Renaissance', 'Impressionism', 'Mannerism Late Renaiss.', 'Minimalism',
                'Naive Art Primitivism', 'New Realism', 'Northern Renaissance', 'Pointillism', 'Pop Art',
                'Post-Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic Cubism', 'Ukiyo-E'],

    "JenAesthetics": ['Baroque', 'Classicism', 'Expressionism', 'Impressionism', 'Mannerism', 'Post-Impression.',
                      'Realism', 'Renaissance', 'Rococo', 'Romanticism', 'Symbolism'],

    "ArTest": ['Abstract Expression.', 'Analytical Cubism', 'Art Nouveau', 'Baroque', 'Color Field Painting',
               'Cubism', 'Early Renaissance', 'Expressionism', 'Fauvism', 'High Renaissance', 'Impressionism',
               'Mannerism Late Renaiss.', 'Minimalism', 'Naive Art Primitivism', 'Neo Classicism',
               'Northern Renaissance', 'Pop Art', 'Post-Impression.', 'Realism', 'Rococo', 'Romanticism',
               'Surrealism', 'Symbolism', 'Synthetic Cubism', 'Ukiyo-E']
}


def add_bars(ax, x, y, y_min, y_max, x_min, x_max, title, x_label, y_label, label="",
             width=0.1, color='', edgecolor='black', linewidth=3):
    ax.bar(x, y, width=width, label=label, color=color, edgecolor=edgecolor, linewidth=linewidth)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    return ax


def load_and_prepare_data(file, pred_field):
    import pandas as pd
    df = pd.read_csv(file)
    preds = df[pred_field].str.lower()
    gt = df["gt"].str.lower()
    return preds, gt


def calculate_metrics(gt, preds, styles):
    recall = recall_score(gt, preds, labels=styles, average=None)
    precision = precision_score(gt, preds, labels=styles, average=None)
    f1 = f1_score(gt, preds, labels=styles, average=None)
    return precision, recall, f1


def adjust_and_save_plot(fig, ax, x, savedir, dataset, title, ylabel, labelsize):
    ax.set_ylabel(ylabel=ylabel, fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x, labels=style_labels[dataset], rotation=45, fontsize=labelsize, horizontalalignment='right')
    ax.legend(loc='upper center', ncol=4, fontsize=23)
    ax.set_title(title, fontsize=25)
    fig.savefig(savedir + "precision_recall_f1score/" + dataset + "_" + ylabel.replace(" ", "_") + "_bars.png",
                bbox_inches='tight')
    fig.savefig(savedir + "precision_recall_f1score/" + dataset + "_" + ylabel.replace(" ", "_") + "_bars.pdf", dpi=100,
                bbox_inches='tight')


def main(dir="results/", dataset="WikiArt", savedir="results/"):
    plt.rcParams['font.family'] = 'Times New Roman'

    models = ["CLIP", "LLaVA", "OpenFlamingo", "GPT-4o"]
    model_names = ["CLIP", "LLaVA", "OpenFlamingo", "GPT-4o"]
    model_colors = ["#ffb000", "#fd6100", "#dc2680", "#785df0"]

    plt.rc('axes', labelsize=30)
    plt.rc('axes', titlesize=30)
    plt.rc('legend', fontsize=30)
    plt.rc('figure', titlesize=25)

    fig1, ax1 = plt.subplots(figsize=(30, 7))
    fig2, ax2 = plt.subplots(figsize=(30, 7))
    fig3, ax3 = plt.subplots(figsize=(30, 7))

    title = dataset + ' Dataset'
    x_min, x_max = 0, 100
    y_min = -0.01
    if dataset == "WikiArt": y_max = 1.03
    if dataset == "JenAesthetics": y_max = 1.01
    if dataset == "ArTest": y_max = 1.19
    bar_width = 0.4
    linewidth = 2
    labelsize = 20

    for i, model in enumerate(models):
        model_dir = dir + "_".join([dataset, "style", model]) + "/"
        if os.path.exists(model_dir + "results_processed.csv"):
            file = model_dir + "results_processed.csv"
            pred_field = "predicted_class"
        else:
            file = model_dir + "results.csv"
            pred_field = "preds"

        preds, gt = load_and_prepare_data(file, pred_field)

        accuracy = (preds == gt).astype(int).sum().item() / len(preds) * 100
        print(f"Accuracy for {model}: {round(accuracy, 2)}%")

        styles = sorted(list(set(gt)))
        x = np.linspace(x_min + 2, x_max - 2, len(styles))

        precision, recall, f1 = calculate_metrics(gt, preds, styles)

        for idx, style in enumerate(styles):
            print(f"Model: {model} | {style} | Precision: {round(precision[idx], 2)} | "
                  f"Recall: {round(recall[idx], 2)} | F1: {round(f1[idx], 2)}")

        # Plotting F1 scores
        ax1 = add_bars(ax1, x + bar_width * i, precision, y_min, y_max, x_min, x_max, title, "", "",
                       label=model_names[i], width=bar_width, color=model_colors[i], linewidth=linewidth)
        ax2 = add_bars(ax2, x + bar_width * i, recall, y_min, y_max, x_min, x_max, title, "", "",
                       label=model_names[i], width=bar_width, color=model_colors[i], linewidth=linewidth)
        ax3 = add_bars(ax3, x + bar_width * i, f1, y_min, y_max, x_min, x_max, title, "", "",
                       label=model_names[i], width=bar_width, color=model_colors[i], linewidth=linewidth)

    adjust_and_save_plot(fig1, ax1, x, savedir, dataset, title, "Precision", labelsize)
    adjust_and_save_plot(fig2, ax2, x, savedir, dataset, title, "Recall", labelsize)
    adjust_and_save_plot(fig3, ax3, x, savedir, dataset, title, "F1 Score", labelsize)


if __name__ == "__main__":
    results_dir = "results/"
    dataset = "ArTest"
    savedir = "results/"

    main(results_dir, dataset, savedir)
