import os
import json
import re

import pandas as pd
import numpy as np
from functools import partial

import torch
import clip

from sklearn.metrics import confusion_matrix

from plots.plots import plot_confusion_matrix

from utils import parse_args, get_style_order


def process_in_batches(data, model, device, batch_size=32):
    embeddings = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        tokenized_batch = clip.tokenize(batch, context_length=77, truncate=True).to(device)
        batch_embeddings = model.encode_text(tokenized_batch).detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def find_word_sentence(text, word):
    # Split the text into sentences
    sentences = text.split('.')
    # Iterate over each sentence
    for sentence in sentences:
        # Check if the sentence contains the word
        if word.lower() in sentence.lower():
            # Return the sentence with the trailing period added back
            return sentence.strip()
    # Return an empty string if no sentence contains the word
    return text


def find_year(text):
    matches = re.findall(r"[0-9]{4}", text)
    if len(matches) == 0:
        return 0
    return matches[0]


def remove_author_expression(text):

    # Multiple names and # 'von'/'van' in last names
    pattern = r'\bby [A-Z][a-z]*(?: [A-Z][a-z]*)*(?: [A-Z]*[a-z]{0,3})*(?: [A-Z][a-z]* )* [A-Z][a-z]* \b'

    # Use re.sub() to remove the matched pattern from the text
    modified_text = re.sub(pattern, '', text)

    # Return the modified text, removing any extra spaces that may result
    return ' '.join(modified_text.split())


def post_process_predictions(preds, model_name):

    # Take into account NaN values in GPT-4o
    preds.loc[preds.isnull()] = ""

    if "ASSISTANT: " in str(preds[0]):
        preds = preds.str.replace("\n", "").str.split("ASSISTANT: ", expand=True)[:][1]
    if "<image>" in str(preds[0]):
        preds = preds.str.split("<image>", expand=True)[3]
    if attribute == "period" or attribute == "art_style" and "GPT-4o" not in model_name:
        if "[author]" in args['clip_prompt']:
            preds = preds.apply(remove_author_expression)
        if "[year]" in args['clip_prompt']:
            preds = preds.apply(remove_year_expression)
        preds = preds.apply(partial(find_word_sentence, word='This painting belongs to'))
        preds = preds.apply(partial(find_word_sentence, word="painting belongs to "))
        preds = preds.apply(partial(find_word_sentence, word="The art style of this painting"))
        preds = preds.apply(partial(find_word_sentence, word='style'))
        preds = preds.apply(partial(find_word_sentence, word='period'))
    if "LLaVa" in model_name and attribute in ["art_style", "period"]:
        preds = preds.apply(process_LLaVa_output_style)
    if "OpenFlamingo" in model_name and attribute in ["art_style", "period"]:
        preds = preds.str.replace("The art period of this painting is", "")
        preds = preds.str.replace("The art style of this painting is", "")
        preds = preds.str.replace("\t", "")
    if "OpenFlamingo" in model_name and attribute == "author":
        preds = preds.apply(partial(find_word_sentence, word="The author of this painting is"))
        preds = preds.str.replace("The author of this painting is ", "")
    if "OpenFlamingo" in model_name and attribute == "year":
        preds = preds.apply(partial(find_word_sentence, word="This painting was made in"))
        preds = preds.apply(replace_century)
        preds = preds.apply(find_year)
        preds = round_year(preds, year_precision)
    if attribute == "year" and "CLIP" not in model_name:
        preds = preds.apply(replace_century)
    if attribute == "year" and "LLaVa" in model_name:
        preds = preds.str.replace("The painting was made in ", "")
        preds = preds.str.replace("the ", "")
    if attribute == "year" and "CLIP" in model_name:
        preds = round_year(preds, year_precision)
    if "GPT-4o" in model_name and dataset_name != "ArTest":
        preds = preds.apply(process_GPT4o_output_style)
    if "GPT-4o" in model_name and dataset_name == "ArTest":
        preds = preds.apply(process_GPT4o_output_style)

    return preds


def process_LLaVa_output_style(preds):
    preds = preds.replace("by William H. Johnson ", "")
    preds = preds.replace("The art style of this painting is ", "")
    preds = preds.replace("The art style of this painting ", "")
    preds = preds.split(", which was ")[:][0]
    preds = preds.split(" which is a ")[:][0]
    preds = preds.split(" which was created ")[:][0]
    preds = preds.split(", which is characterized")[:][0]

    preds = preds.replace("the art style of ", "")
    preds = preds.split("painting belongs to")[:][-1]
    preds = preds.split(" style")[:][0]
    preds = preds.split(" period")[:][0]
    preds = preds.split(" era, as ")[:][0]
    preds = preds.replace("art", "").replace("the", "")
    preds = preds.replace("characterized by a combination of ", "")
    preds = preds.replace("a combination of ", "")
    preds = preds.replace("can be described as a mix of ", "")
    preds = preds.replace("considered to be ", "")
    preds = preds.replace("by André Bauchant is ", "")
    preds = preds.replace("known as ", "")
    preds = preds.replace("a mix of ", "")
    return preds


def process_GPT4o_output_style(text):
    text = find_word_sentence(text, "a notable example of Venetian Renaissance art")
    text = find_word_sentence(text, "This painting belongs to the ")
    text = find_word_sentence(text, "The painting belongs to the ")
    text = find_word_sentence(text, "The painting appears to belong to the")
    text = find_word_sentence(text, "is an example of ")
    text = find_word_sentence(text, "belongs to the ")
    text = find_word_sentence(text, "is an art movement")
    text = find_word_sentence(text, "period")
    text = find_word_sentence(text, "art style")
    text = find_word_sentence(text, "example of ")

    text = text.split(" is an art style that")[0]
    text = text.split("is an example of ")[-1]
    text = text.split(" is an art movement")[0]
    text = text.split("belongs to the ")[-1]
    text = text.split("appears to belong to the ")[-1]
    text = text.split("This painting belongs to the art style of ")[-1]
    text = text.split("The painting you provided belongs to the ")[-1]
    text = text.split("This painting belongs to the ")[-1]
    text = text.split("This painting is characteristic of the ")[-1]
    text = text.split("painting appears to belong to the ")[-1]
    text = text.split(", which is")[0]
    text = text.split("example of the ")[-1]
    text = text.split("a notable example of ")[-1]
    text = text.split(", typical of the ")[-1]
    text = text.split("The painting in question is The Tempest by Giorgione, a notable example of ")[-1]
    text = text.split("The painting you provided is a classic example of ", )[-1]

    text = text.replace("This painting is characteristic of the ", "")
    text = text.replace("This painting exhibits characteristics of the ", "")
    text = text.replace("This painting is associated with the", "")
    text = text.replace("This drawing, dated 1920, bears a ", "")
    text = text.replace('The painting in question is The Tempest by Giorgione, a notable example of ', "")
    text = text.replace("The painting you have posted belongs to ", "")
    text = text.replace(", a phase that lasted from 1901 to 1904", "")
    text = text.replace("This painting appears to belong to the ", "")

    text = text.replace("art style known as ", "").replace("art style", "").replace("**", "").replace("\"", "")

    return text


def remove_year_expression(text):

    pattern = r"\bmade in \d{4} \b"

    # Use re.sub() to remove the matched pattern from the text
    modified_text = re.sub(pattern, '', text)

    # Return the modified text, removing any extra spaces that may result
    return ' '.join(modified_text.split())


def replace_century(text):
    """ When predicting the year, transforms strings like 'The painting was made in the 16th century.'
    into 'The painting was made in 1500'"""

    def century_to_year(match):
        century = int(match.group(1))
        year = (century - 1) * 100
        return f"{year}"

    # Regex pattern to match centuries (e.g., 16th, 21st, 22nd)
    pattern = re.compile(r'(\d+)(?:st|nd|rd|th) century')

    # Check if the text contains the pattern and perform the substitution if it does
    if pattern.search(text):
        new_text = pattern.sub(century_to_year, text)
        return new_text
    else:
        return text


def round_year(years, precision=100):
    gt_numeric = pd.to_numeric(years, errors='coerce')
    gt_numeric.fillna(0, inplace=True)  # Replace NaN values with 0
    return pd.to_numeric(np.floor(gt_numeric / precision) * precision).astype('Int64').astype(str)


def main(dataset_name="JenAethetics", results_dir="results/JenAethetics/LLaVa/5/",
         prompt="[ART_STYLE]", attribute="style", year_precision=100, plot_conf_matrix=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with open(results_dir + "args.json", "r") as f:
        args = json.load(f)

    print(args['dataset_name'], end="\t")
    print(args['annotation_file'], end="\t")
    print(args['clip_prompt'], end="\t")
    print(args['attribute'], end="\t")

    results_file = results_dir + "results.csv"
    results = pd.read_csv(results_file)

    if attribute == "style" or attribute == "period":
        gt_classes = get_style_order(dataset_name)
    elif attribute == "author":
        gt_classes = list(set(results["gt"].values.tolist())) + ["unknown"]
    elif attribute == "year":
        start = 1400 if dataset_name == "WikiArt" else 1400
        gt_classes = [str(i) for i in range(start, 2100, year_precision)]
    else:
        gt_classes = list(set(results["gt"].values.tolist()))

    # Possible classes
    classes = [prompt.replace("[" + attribute + "]", style.replace("_", " ").lower()) for style in gt_classes]

    # Extract clip text embeddings for the ground-truth labels
    class_embeddings = process_in_batches(classes, model, device, batch_size=32)
    class_embeddings_norm = np.linalg.norm(class_embeddings, axis=1)

    # Post-process the predictions
    preds = results["preds"]
    preds = post_process_predictions(preds, results_file)

    # Extract CLIP text embeddings from the predictions and compare them to the ground-truth
    preds_embeddings = process_in_batches(preds.tolist(), model, device)
    preds_embeddings_norm = np.linalg.norm(preds_embeddings, axis=1)
    norm_matrix = np.outer(preds_embeddings_norm, class_embeddings_norm.T)
    dot = preds_embeddings @ class_embeddings.T
    cosine_similarity = dot / norm_matrix

    predicted_class_id = np.argmax(cosine_similarity, axis=1)
    predicted_class = [gt_classes[i].replace("_", " ").lower() for i in predicted_class_id]

    gt = results["gt"]
    if attribute == "art_style" or attribute == "period":
        gt = np.array([GT.replace("_", " ").lower() for GT in gt.values.tolist()])
    if attribute == "year":
        gt = round_year(gt, year_precision)

    accuracy = (predicted_class == gt).astype(int).sum().item() / len(predicted_class) * 100
    accuracy = " " + str(round(accuracy, 2))
    print(f"Accuracy on", dataset_name, "is", accuracy)

    results["predicted_class"] = predicted_class
    results.to_csv(results_dir + "results_processed.csv")

    if plot_conf_matrix:
        conf_matrix = confusion_matrix(gt, results["predicted_class"],
                                       labels=[s.replace("_", " ").lower() for s in gt_classes])
        plot_confusion_matrix(conf_matrix, results_dir + "confusion_matrix.png",
                              labels=[s.replace("_", " ").lower()
                                      for s in gt_classes], accuracy=str(accuracy))

    return classes, preds, cosine_similarity


if __name__ == "__main__":
    args = parse_args()
    print(args)

    dataset_name = args.dataset_name
    results_dir = args.results_dir
    prompt = args.clip_prompt
    year_precision = args.year_precision
    attribute = args.attribute
    plot_conf_matrix = args.plot_conf_matrix

    classes, preds, cosine_similarity = main(dataset_name, results_dir, prompt, attribute,
                                             year_precision=year_precision, plot_conf_matrix=plot_conf_matrix)



