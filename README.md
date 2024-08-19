# Have Large Vision-Language Models Mastered Art History?
This repository accompanies the paper "Have Large Vision-Language Models Mastered Art History?".

The emergence of large Vision-Language Models (VLMs) has recently established new baselines in image classification across multiple domains. However, the performance of VLMs in the specific task of artwork classification, particularly art style classification of paintings — a domain traditionally mastered by art historians — has not been explored yet. Artworks pose a unique challenge compared to natural images due to their inherently complex and diverse structures, characterized by variable compositions and styles. Art historians have long studied the unique aspects of artworks, with style prediction being a crucial component of their discipline. This paper investigates whether large VLMs, which integrate visual and textual data, can effectively predict the art historical attributes of paintings. We conduct an in-depth analysis of four VLMs, namely CLIP, LLaVA, OpenFlamingo, and GPT-4o, focusing on zero-shot classification of art style, author and time period using two public benchmarks of artworks. Additionally, we present ArTest, a well-curated test set of artworks, including pivotal paintings studied by art historians.


## VLMs inference 


### CLIP


`conda env create -f clip_environment.yml`

Example: Art style prediction on ArTest 
`python CLIP.py  --dataset_name ArTest --image_folder_path datasets/ArTest/ 
    --annotation_file datasets/ArTest/ArTest.csv --features_dir datasets/ArTest/CLIP_features/ 
    --clip_prompt the\ art\ style\ of\ the\ painting\ is\  --image_model ViT-B/32 --attribute art_style --n -1`


### LLaVA

`conda env create -f llava_environment.yml`

The code runs on GPU.


### OpenFlamingo 

`conda env create -f openflamingo_environment.yml`

The code runs on GPU.

### GPT-4o

Utilizes the Open API. Since the code used the API, it does not require a dedicated conda environment.  
To use the API, you need a OpenAI key. Please create and copy your key in a OpenAI_key.txt file.

