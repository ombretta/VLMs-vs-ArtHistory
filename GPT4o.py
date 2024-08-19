import os
import pandas as pd
import base64
import requests

from utils import parse_args, save_args_to_file


def predict_image(image_base64, headers, prompt="Describe this image."):

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    try:
        response_data = response.json()
        print(response_data)  # Log the full response for debugging

        if 'choices' in response_data:
            pred = response_data["choices"][0]["message"]['content']
            return pred
        else:
            print("Error: 'choices' not in response:", response_data)
            return None
    except ValueError as e:
        print("Error parsing response as JSON:", e)
        print("Response content:", response.content)
        return None

    return pred


def main(res_dir, image_folder_path, annotation_file, prompt, n, attribute="style"):

    # OpenAI API Key
    with open('OpenAI_key.txt', 'r') as f:
        api_key = f.read().replace("\n", "")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Load dataset and labels
    annotations = pd.read_csv(annotation_file)
    images = annotations["file"].values.tolist()
    print(annotations[attribute])

    styles = [s.replace("_", " ").lower() for s in annotations[attribute].values.tolist()]
    preds = []

    if n == -1:
        n = len(images)

    for idx, (img, style) in enumerate(zip(images[:n], styles[:n])):
        with open(os.path.join(image_folder_path, img), 'rb') as image_file:
            image_bytes = image_file.read()

            # Encode the image in base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            pred = predict_image(image_base64, headers, prompt)
            preds.append(pred)

            print(idx, img, "- GT:", style, "Pred:", pred)

    pd.DataFrame({'images': images[:n], "gt": styles[:n], "preds": preds}).to_csv(os.path.join(res_dir, "results.csv"))


if __name__ == "__main__":
    args = parse_args()
    print(args)

    res_dir, i = os.path.join("results", args.dataset_name, "GPT-4o"), 1
    while os.path.exists(os.path.join(res_dir, str(i))):
        i += 1
    res_dir = os.path.join(res_dir, str(i))
    os.mkdir(res_dir)

    save_args_to_file(args, os.path.join(res_dir, "args.json"))

    main(res_dir, args.image_folder_path, args.annotation_file, args.clip_prompt, args.n, args.attribute)
