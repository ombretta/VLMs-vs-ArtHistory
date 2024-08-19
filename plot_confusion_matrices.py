import os
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from plots.plots import plot_confusion_matrix

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


def load_and_prepare_data(file, pred_field):
    import pandas as pd
    df = pd.read_csv(file)
    preds = df[pred_field].str.lower().str.replace(" ", "_")
    gt = df["gt"].str.lower().str.replace(" ", "_")
    return preds, gt


def main(dir="results/", dataset="WikiArt", savedir="results/"):
    plt.rcParams['font.family'] = 'Times New Roman'

    models = ["CLIP", "LLaVa", "OpenFlamingo", "GPT-4o"]
    model_names = ["CLIP", "LLaVA", "OpenFlamingo", "GPT-4o"]

    plt.rc('axes', labelsize=30)
    plt.rc('axes', titlesize=30)
    plt.rc('legend', fontsize=30)
    plt.rc('figure', titlesize=25)

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

        conf_matrix = confusion_matrix(gt, preds, labels=styles)
        cmap = plt.cm.Blues
        title = model_names[i] + " (Acc. " + str(round(accuracy, 2)) + "%)"
        figpath = savedir + "confusion_matrices/" + dataset + "_" + model + "_confusion.pdf"
        plot_confusion_matrix(conf_matrix, figpath, figsize=(35, 20),
                              labels=(style_labels[dataset], style_labels[dataset]),
                              title=title, accuracy=round(accuracy, 2), cmap=cmap, xticks_rotation=45)


if __name__ == "__main__":
    results_dir = "results/"
    dataset = "JenAesthetics"
    savedir = "results/"

    main(results_dir, dataset, savedir)
