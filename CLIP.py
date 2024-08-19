import os
import json
import wandb

import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image

from sklearn.metrics import confusion_matrix

from plots.plots import plot_confusion_matrix

from utils import parse_args


def save_args_to_file(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)


def main(res_dir, image_folder_path, annotation_file, features_dir, clip_prompt, image_model, n,
         attribute="style", save_features=False):

    Image.MAX_IMAGE_PIXELS = 1000000000  # Adjust this value as needed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(image_model, device=device)

    annotations = pd.read_csv(annotation_file)
    images = annotations["file"].values.tolist()

    print(annotations[attribute].values.tolist())
    styles = [str(s).replace("_", " ").lower() for s in annotations[attribute].values.tolist()]
    classes = sorted(list(set(styles)))
    preds = []
    correct = 0

    if n == -1:
        n = len(images)

    for idx, (img, style) in enumerate(zip(images[:n], styles[:n])):

        image = preprocess(Image.open(image_folder_path + img)).unsqueeze(0).to(device)

        original_prompt = clip_prompt

        print("clip_prompt", clip_prompt, "annotations.keys()", annotations.keys())
        if "[author]" in clip_prompt and "author" in annotations.keys():
            print(idx, annotations.loc[idx]["author"])
            clip_prompt = clip_prompt.replace("[author]", annotations.loc[idx]["author"].replace("_", " "))
            print(clip_prompt)

        if "[year]" in clip_prompt and "year" in annotations.keys():
            clip_prompt = clip_prompt.replace("[year]", annotations.loc[idx]["year"])
            print(clip_prompt)

        prompts = [clip_prompt + c for c in classes]
        text = clip.tokenize(prompts).to(device)

        clip_prompt = original_prompt

        with torch.no_grad():

            if save_features:
                image_features = model.encode_image(image)
                torch.save(image_features, features_dir+img.split("/")[-1].replace(".jpg", ".pt"))

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        preds.append(classes[np.argmax(probs)])
        if style == classes[np.argmax(probs)]:
            correct = correct + 1

        print(idx, img, "- GT:", style, "Pred:", classes[np.argmax(probs)])

    accuracy = " (Acc. : %.2f%%)" % round((correct / len(images)) * 100, 2)

    conf_matrix = confusion_matrix(styles[:n], preds, labels=classes)
    plot_confusion_matrix(conf_matrix, res_dir+"confusion_matrix.png", labels=classes, accuracy=accuracy)

    pd.DataFrame({'images': images[:n], "gt": styles[:n], "preds": preds}).to_csv(res_dir+"results.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    res_dir, i = "results/" + args.dataset_name + "/CLIP/", 1

    while os.path.exists(res_dir+str(i)):
        i += 1
    res_dir += str(i) + "/"
    os.mkdir(res_dir)

    save_args_to_file(args, res_dir+"args.json")

    main(res_dir, args.dataset_name, args.image_folder_path, args.annotation_file, args.features_dir,
         args.clip_prompt, args.image_model, args.n, args.attribute, args.save_features)

