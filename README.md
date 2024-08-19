# Have Large Vision-Language Models Mastered Art History?

## Introduction

This repository accompanies the paper "Have Large Vision-Language Models Mastered Art History?".

The emergence of large Vision-Language Models (VLMs) has recently established new baselines in image classification across multiple domains. However, the performance of VLMs in the specific task of artwork classification, particularly art style classification of paintings — a domain traditionally mastered by art historians — has not been explored yet. Artworks pose a unique challenge compared to natural images due to their inherently complex and diverse structures, characterized by variable compositions and styles. Art historians have long studied the unique aspects of artworks, with style prediction being a crucial component of their discipline. This paper investigates whether large VLMs, which integrate visual and textual data, can effectively predict the art historical attributes of paintings. We conduct an in-depth analysis of four VLMs, namely CLIP, LLaVA, OpenFlamingo, and GPT-4o, focusing on zero-shot classification of art style, author and time period using two public benchmarks of artworks. Additionally, we present ArTest, a well-curated test set of artworks, including pivotal paintings studied by art historians.


## Project Structure
The repository contains the code to reproduce the main results from the paper, as well as the data and results collected in our project and the new ArTest test set, curated by art historians.

### Directories

- **`datasets/`**: This folder contains the datasets and data files used in the project.
  - `WikiArt/`: Please download and place WikiArt images and annotations in this folder. The WikiArt datasets can be found [here](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md). 
  - `JenAesthetics/`: Please download and place JenAesthetics images and annotations in this folder. The JenAesthetics datasets can be found [here](https://github.com/Bin-ary-Li/JenAesthetics?tab=readme-ov-file).
  - `ArTest/`: ArTest is a test set of 147 painting images, well curated by art historians. The folder contains all the ArTest images and annotations collected in this project.

- **`results/`**: This directory contains all the results collected in this project. The subdirectories contain the results for each tested datasets and VLM, on the different prediction tasks. The folder names specify the test: [dataset_name]\_[predicted_attribute]\_[VLM]. Each folder contains a `results.csv` file, with the zero-shot predictions for each painting. In addition, the test argument parameters are reported in the `args.json` file. In some cases, we include the confusion matrix of the predictions (`confusion_matrix.png`).    


## VLMs inference 

The prediction task is specified through the `--prompt` argument (e.g., `--prompt 
"To which art style does this painting belong?"`) and the `--attribute` argument (art_style, author, year). 

The results are saved in the `results/` folder.

### CLIP

Based on the [official CLIP implementation](https://github.com/openai/CLIP). Requirements installation through:

`conda env create -f clip_environment.yml`

Example of usage: Art style prediction on ArTest 

    conda activate clip
    python CLIP.py  --dataset_name ArTest --image_folder_path datasets/ArTest/ --annotation_file datasets/ArTest/ArTest.csv --features_dir datasets/ArTest/CLIP_features/ --prompt the\ art\ style\ of\ the\ painting\ is\  --image_model ViT-B/32 --attribute art_style --n -1


### LLaVA

We utilize the [HuggingFace implementation](https://huggingface.co/llava-hf/llava-1.5-7b-hf), based on the `transformers` library. Requirements installation through:

`conda env create -f llava_environment.yml`

Please note: The code runs only on GPU.

Example of usage: Art style prediction on ArTest 

    conda activate llava
    python LLaVA.py --dataset_name ArTest --image_folder_path datasets/ArTest/ --annotation_file datasets/ArTest/ArTest.csv --prompt "To which art style does this painting belong?" --n -1 --attribute art_style


### OpenFlamingo 

We utilize the [HuggingFace implementation](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b). Requirements installation through:

`conda env create -f openflamingo_environment.yml`

Please note: The code runs only on GPU.

Example of usage: Art style prediction on ArTest 
    
    conda activate openflamingo 
    python OpenFlamingo.py --dataset_name ArTest --image_folder_path datasets/ArTest/ --annotation_file datasets/ArTest/ArTest.csv --attribute art_style --prompt "The art style of this painting is " --n -1 --path_example_image1 figures/OpenFlamingo_examples/Baroque_deHeem_Flowers_1660.jpg --path_example_image2 figures/OpenFlamingo_examples/Pointillism_Signac_CapoNoli_1898.jpg --max_new_tokens 20 --text_model anas-awadalla/mpt-1b-redpajama-200b-dolly



### GPT-4o

Utilizes the [Open API](https://platform.openai.com/docs/overview). Since the code uses the API, it does not require a dedicated conda environment.  
To use the API, you need a OpenAI key. Please create and copy your key in a OpenAI_key.txt file.

Example of usage: Art style prediction on ArTest
    
    python GPT4o.py --dataset_name ArTest --image_folder_path datasets/ArTest/ --annotation_file datasets/ArTest/ArTest.csv --clip_prompt "To which art style does this painting belong?" --n -1 --attribute art_style



## Evaluation

To evaluate the zero-shot predictions by the text generation VLMs (LLaVA, OpenFlamingo and GPT_4o), we post-process the VLMs outputs and compare them with the ground truth art historical attributes. This is done through [zeroshot_eval.py](zeroshot_eval.py), which uses the CLIP text encoder.
For CLIP, this prost-processing step is not needed: this model directly assigns one predicted class to the painting. 

### Results post-processing: 

Example:
```
conda activate clip
python zeroshot_eval.py --dataset_name WikiArt --results_dir results/WikiArt/LLaVA/WikiArt_style_LLaVa/ --prompt "[art_style]" --attribute art_style --plot_conf_matrix True 
```

### Precision and Recall
To plot precision and recall for the prediction of each art style on one specific dataset, run `plot_precision_recall.py`. For example:

```
python plot_precision_recall.py --dataset_name WikiArt --results_dir results/ 
```

### Confusion Matrix
To plot the confusion matrices for the style predictions on one specific dataset, run `plot_precision_recall.py`. For example:

```
python plot_precision_recall.py  --dataset_name WikiArt --results_dir results/ 
```