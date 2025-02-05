# Webiks-Hebrew-RAGbot-Trainer

This repository contains code for fine-tuning an [me5-large](https://huggingface.co/intfloat/multilingual-e5-large) model using the [sentence-transformers](https://huggingface.co/sentence-transformers) library, specifically tailored for Q\&A Retrieval on the [Kol-Zchut](http://kolzchut.org.il/) website pages. However, the code can be adapted for various datasets, or even different models within the sentence-transformers framework.      

## Environment

Create a conda environment using: `conda env create -f environment.yml`

## Data
In order to be able to train a model on your machine you must download the following files from our drive.
### Kol-Zchut Corpus

* The file: [`Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json`](https://drive.google.com/file/d/18hAihDl0NlBz4EFubSnN7YMwL4v58qP_/view?usp=drive_link) is the paragraph corpus used to train our model. It contains all relevant paragraphs from the Kol-Zchut website, extracted by splitting the Kol-Zchut webpages according to their HTML titles, and then combining paragraphs up to the maximum context size of the me5-large model, which is 512 tokens.
* The corpus was extracated from the KolZchut website in May 2024 and is not necessarily up to date with today's website.
* For more information about this file you can check this dedicated [repo](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-Paragraph-Corpus).

### Training Data

* The file: [`Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv`](https://drive.google.com/file/d/18WE5JARjzBkBD9kCd7cTxm1-7XX4-ylG/view?usp=drive_link) contains our training set of questions and corresponding answers. The answers consist of relevant paragraphs to each question taken from the paragraph corpus.
* For more information about this file you can check this dedicated [repo](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-KolZchut-QA-Training-DataSet).

## Training

* You can run training using the following example command: `python3 run.py --train_path path/to/Webiks_Hebrew_RAGbot_KolZchut_QA_Training_DataSet_v0.1.csv --corpus_path path/to/Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json`. You can provide your own validation set using the `--val_path` argument. If you do not your train set will be split to train and validation according to the `--val_size` argument.  
* This will output the model in `models/Webiks_KolZchut_QA_Embedder` and the results on you validation set in the file `ranking_results.csv`. Note that you can use the `model_version` argument if you wish to manage various versions of models.  
* The default parameters are set for training on an NVIDIA GeForce RTX 3090 with 24gb of GPU memory. You can, of course, adjust the batch size as needed.

## Models

* After training, the model will be saved to the location `models/Webiks_KolZchut_QA_Embedder`. You can now use this model for inference on an entire validation set using this repo or learn how to integrate this model for inference in a production environment using our [RAG Framework](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-Demo).  
* The fine-tuned model resulting from the training can be downloaded from our [drive](https://drive.google.com/file/d/1eFAddJWBWDvoid-Gyn6ZT5jPwf-vNPI8/view?usp=drive_link).

## Evaluation

* You can run evaluation on any dataset using this example command: `python3 run.py --eval --val_path path/to/your_val_set.csv --use_artifact --corpus_path path/to/Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json`. The format of your validation set should be similar to the provided train set.
* This will output the ranking of documents on the evaluation set in `results/ranking_results.csv`. Additionally, it will write a line in the csv `results/Information-Retrieval_evaluation_results.csv` for each model you run.
