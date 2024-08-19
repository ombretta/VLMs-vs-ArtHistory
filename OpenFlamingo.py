'''GPU only'''

import os
import pandas as pd
import torch

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from PIL import Image

from utils import parse_args, save_args_to_file


def main(res_dir, dataset_name, image_folder_path, annotation_file, n, attribute="style",
         clip_prompt="The art style of this painting is ",
         path_example1="Cubism_Picasso_Avignon_1907.jpg", style_example1="Cubism",
         path_example2="Realism_David_Marat_1793.jpg", style_example2="Realism",
         max_new_tokens=200, LLM_id="anas-awadalla/mpt-1b-redpajama-200b"):

    # Deal with large images
    Image.MAX_IMAGE_PIXELS = 1000000000

    # Load model
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=LLM_id,
        tokenizer_path=LLM_id,
        cross_attn_every_n_layers=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to GPU 0
    model = model.to(device)

    checkpoint_path = "openflamingo/OpenFlamingo-9B-vitl-mpt7b" if LLM_id in ["anas-awadalla/mpt-7b"] \
        else "openflamingo/OpenFlamingo-3B-vitl-mpt1b"

    checkpoint_path = hf_hub_download(checkpoint_path, "checkpoint.pt", cache_dir=cache_dir)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    # Load dataset and labels
    annotations = pd.read_csv(annotation_file)
    images = annotations["file"].values.tolist()
    print(annotations[attribute])

    styles = [str(s).replace("_", " ").lower() for s in annotations[attribute].values.tolist()]
    preds = []

    # Load example images
    example_image1 = Image.open(path_example1)
    example_image2 = Image.open(path_example2)

    # Define prompt
    prompt = ("<image>" + clip_prompt + style_example1 + ".<|endofchunk|>" + "<image>" + clip_prompt + style_example2 +
              ".<|endofchunk|>" + "<image>" + clip_prompt)

    print("OpenFlamingo prompt:", prompt)
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(
        [prompt],
        return_tensors="pt",
    )

    if n == -1:
        n = len(images)

    for idx, (img, style) in enumerate(zip(images[:n], styles[:n])):

        # Preprocessing images
        query_image = Image.open(image_folder_path + img)
        vision_x = [image_processor(example_image1).unsqueeze(0), image_processor(example_image2).unsqueeze(0),
                    image_processor(query_image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vision_x = vision_x.to(device)

        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to(device),
            attention_mask=lang_x["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            num_beams=3,
        )

        pred = tokenizer.decode(generated_text[0])
        preds.append(pred)
        print(idx, img, "- GT:", style, "Pred:", pred)

    pd.DataFrame({'images': images[:n], "gt": styles[:n], "preds": preds}).to_csv(res_dir + "results.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    res_dir, i = "results/" + args.dataset_name + "/OpenFlamingo/", 1
    while os.path.exists(res_dir + str(i)):
        i += 1
    res_dir += str(i) + "/"
    os.mkdir(res_dir)

    save_args_to_file(args, res_dir + "args.json")

    LLM_id = args.text_model

    path_example1 = args.path_example_image1
    path_example2 = args.path_example_image2

    if args.attribute == "art_style" or args.attribute == "period":
        style_example1 = path_example1.split("_")[0]
        style_example2 = path_example2.split("_")[0]
    elif args.attribute == "author":
        style_example1 = path_example1.split("_")[1]
        style_example2 = path_example2.split("_")[1]
    elif args.attribute == "year":
        style_example1 = path_example1.split("_")[-1].split(".jpg")[0]
        style_example2 = path_example2.split("_")[-1].split(".jpg")[0]

    main(res_dir, args.image_folder_path, args.annotation_file, args.n, args.attribute, args.clip_prompt,
         path_example1, style_example1, path_example2, style_example2, args.max_new_tokens, LLM_id)

