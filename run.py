# This code is licensed under the MIT License.
# See the LICENSE file for more details.

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses
from sentence_transformers.util import cos_sim
import argparse
import os
from utils import UniqueQuestionSampler, CustomInformationRetrievalEvaluator, load_model, load_val_set, split_train_eval

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(model, train_path, val_path, val_size, batch_size, epochs, lr, weight_decay,
          warmup_steps, output_path, similarity_scaler,
          corpus_path, ranking_output_name, corpus_text_column='content'):
    """
    Trains a retrieval model using the provided training and validation data.

    This function sets up the training environment, loads and preprocesses data,
    initializes an evaluator, and trains the model using MultipleNegativesRankingLoss.
    It also performs evaluation during training and saves the best model.
    """   
    ranking_csv_path=os.path.join(output_path, ranking_output_name)
    
    model.train()
 
    df = pd.read_csv(train_path, encoding='utf-8')    
    if val_path:
        df_val = pd.read_csv(val_path, encoding='utf-8')
        df_train = df
    else:
        df_train, df_val = split_train_eval(df, val_size)
        
    queries, corpus, relevant_docs, corpus_df = load_val_set(df_val, corpus_text_column, corpus_path=corpus_path)
        
    evaluator = CustomInformationRetrievalEvaluator(queries=queries, corpus=corpus,
                                                    relevant_docs=relevant_docs,
                                                    corpus_df=corpus_df,
                                                    ranking_csv_path=ranking_csv_path,
                                                    mrr_at_k=[1,3,5,10], ndcg_at_k=[1,5], map_at_k=[1,5],
                                                    corpus_chunk_size=1000,
                                                    show_progress_bar=True, batch_size=4, write_csv=True,
                                                    main_score_function='cos_sim', score_functions={'cos_sim': cos_sim},
                                                    hit_at_k=[1, 3, 5])
    

    train_data = [InputExample(texts=[row['question'], row['paragraph']]) for _, row in df_train.iterrows()]
        
    sampler = UniqueQuestionSampler(train_data, batch_size)
    train_dataloader = DataLoader(train_data,
                                num_workers=os.cpu_count(), pin_memory=True, batch_sampler=sampler)
    
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=similarity_scaler)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, 
              evaluator=evaluator, warmup_steps=warmup_steps, scheduler = 'WarmupLinear',
              optimizer_params={'lr': lr, 'weight_decay': weight_decay}, output_path=output_path, save_best_model=True, 
              show_progress_bar=True)


def eval(model_name, model_version, csv_path, use_artifact, output_path, 
        returned_corpus_col, ranking_output_name, corpus_path, corpus_text_column='content'):
    """
    Evaluates a retrieval model using a specified evaluation dataset.

    This function loads a pre-trained model, sets up an evaluation environment,
    and performs evaluation. It processes the evaluation data, runs the model on the test set,
    and saves the evaluation results.
    """
    ranking_csv_path=os.path.join(output_path, ranking_output_name)
    
    df_val = pd.read_csv(csv_path, encoding='utf-8')
        
    queries, corpus, relevant_docs, corpus_df = load_val_set(df_val, corpus_text_column, corpus_path=corpus_path)
    
    evaluator = CustomInformationRetrievalEvaluator(queries=queries, corpus=corpus,
                                                    relevant_docs=relevant_docs,
                                                    corpus_df=corpus_df,
                                                    ranking_csv_path=ranking_csv_path,
                                                    mrr_at_k=[1,3,5,10,20,200], ndcg_at_k=[1,5], map_at_k=[1,5,40],
                                                    corpus_chunk_size=1000,
                                                    show_progress_bar=True, batch_size=4, write_csv=True,
                                                    main_score_function='cos_sim', score_functions={'cos_sim': cos_sim},
                                                    returned_corpus_column=returned_corpus_col, k=20, hit_at_k=[1, 3, 5])
    
    model = load_model(model_name, model_version, use_artifact)
    
    model.eval()
    evaluator(model, output_path=output_path)
    
    results_csv = os.path.join(output_path, 'Information-Retrieval_evaluation_results.csv')
    df = pd.read_csv(results_csv)
    df.at[df.index[-1], 'model_name'] = model_name
    df.at[df.index[-1], 'model_version'] = model_version
    
    df.to_csv(results_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true', help='If true wil run evaluation. If fase, will run training.')
    parser.add_argument("--train_path", type=str, help='Path to training set (csv file)')
    parser.add_argument("--val_path", type=str, default=None, help='Path to validation set (csv file) this is optional, and if not provided the train set will be split')
    parser.add_argument("--val_size", type=float, default=0.1, help='Float from 0 to 1. The percentage size of the validation set.')
    parser.add_argument("--output_path", default='results', type=str, help='path to outputs, will output model in case of running training, will output evaluation results in case of running eval')
    parser.add_argument("--ranking_output_name", type=str, default='ranking_results.csv', help='name for the output ranking csv (evaluation only)')
    parser.add_argument("--batch_size", type=int, default=4, help='Training batch size')
    parser.add_argument("--epochs", type=int, default=10, help='Training Epochs')
    parser.add_argument("--lr", type=float, default=0.00001, help='Training optimizer learning rate')
    parser.add_argument("--weight_decay", type=float, default=0.01, help='Training weight decay')
    parser.add_argument("--similarity_scaler", type=float, default=100, help='Similarity score for sentence_transformers.losses.MultipleNegativesRankingLoss')
    parser.add_argument("--warmup_steps", type=int, default=243, help='Training phase warmup steps.')
    parser.add_argument("--use_artifact", action='store_true', help='Whether to load a model from storage or use pretrained version from huggingface')
    parser.add_argument("--model_version", type=str, default='', help='Can be used to handle multiple versions of models. If stated, will save model with a specific version name in training and will know to load it in evaluation.') 
    parser.add_argument("--corpus_path", type=str, help='Corpus file containing all of the text that can be retrieved. In our case, all of the paragraphs.')
    parser.add_argument("--returned_corpus_col", type=str, default='link', help='Column in the corpus that is returned for output. Can also be changed to doc_id')

    args = parser.parse_args()
    model_name = 'Webiks_KolZchut_QA_Embedder'
    if not args.eval:
        model = load_model(model_name, args.model_version, args.use_artifact)      
        output_path = f'models/{model_name}:{args.model_version}' if args.model_version else f'models/{model_name}/'
        os.makedirs(output_path, exist_ok=True)
        
        train(model, args.train_path, args.val_path, args.val_size, args.batch_size, 
            args.epochs, args.lr, args.weight_decay, warmup_steps=args.warmup_steps,
            output_path=output_path, similarity_scaler=args.similarity_scaler,
            corpus_path=args.corpus_path, ranking_output_name=args.ranking_output_name)
    else:
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)
        eval(model_name, args.model_version, args.val_path, args.use_artifact,
             output_path, returned_corpus_col=args.returned_corpus_col, ranking_output_name=args.ranking_output_name, 
             corpus_path=args.corpus_path)
