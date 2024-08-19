'''GPU only'''

import os
import requests

import pandas as pd

from PIL import Image

import torch
from transformers import BitsAndBytesConfig, pipeline

from utils import parse_args, save_args_to_file


def main(res_dir, image_folder_path, annotation_file, prompt, n, attribute="style",
         max_new_tokens=200):
    Image.MAX_IMAGE_PIXELS = 1000000000

    # Load model
    model_id = "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        load_in_8bit=False
    )
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

    # Define prompt
    prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"

    # Load dataset and labels
    annotations = pd.read_csv(annotation_file)
    images = annotations["file"].values.tolist()
    print(annotations[attribute])

    styles = [str(s).replace("_", " ").lower() for s in annotations[attribute].values.tolist()]
    preds = []

    if n == -1:
        n = len(images)

    for idx, (img, style) in enumerate(zip(images[:n], styles[:n])):

        original_prompt = prompt

        if "[author]" in prompt and "author" in annotations.keys():
            print(idx, annotations["author"].loc[idx], img)
            prompt = prompt.replace("[author]", annotations["author"].loc[idx].replace("_", " "))
            print(prompt)

        if "[year]" in prompt and "year" in annotations.keys():
            prompt = prompt.replace("[year]", str(annotations["year"].loc[idx]))
            print(prompt)

        image = Image.open(image_folder_path + img)
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
        pred = outputs[0]['generated_text']
        preds.append(pred)

        prompt = original_prompt

        print(idx, img, "- GT:", style, "Pred:", pred)

    pd.DataFrame({'images': images[:n], "gt": styles[:n], "preds": preds}).to_csv(res_dir + "results.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    res_dir, i = "results/" + args.dataset_name + "/LLaVa/", 1
    while os.path.exists(res_dir + str(i)):
        i += 1
    res_dir += str(i) + "/"
    os.mkdir(res_dir)

    save_args_to_file(args, res_dir + "args.json")

    main(res_dir, args.image_folder_path, args.annotation_file,
         args.prompt, args.n, args.attribute, args.max_new_tokens)
