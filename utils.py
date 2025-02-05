# This code is licensed under the MIT License.
# See the LICENSE file for more details.

import pandas as pd
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import AutoTokenizer
import csv
import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
    
class CustomInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    Custom evaluator for information retrieval tasks, extending InformationRetrievalEvaluator.

    This class provides enhanced evaluation capabilities for retrieval models:

    Computes custom metrics based on document relevance from paragraphs
    Writes detailed evaluation results to a CSV file, including:
    1. Query text
    2. Top-k retrieved documents with scores
    3. Relevant documents and their positions
    4. Hit@k metrics for specified k values
    5. Handles document-paragraph mapping
    """
    def __init__(self, corpus, queries, relevant_docs, max_tokens_paragraph=512, corpus_df=None, 
                 ranking_csv_path='evaluation_results.csv',
                 returned_corpus_column='link', k=5, hit_at_k=[1, 3, 5], *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.max_tokens_paragraph = max_tokens_paragraph
        self.corpus_df = corpus_df
        self.ranking_csv_path = ranking_csv_path
        self.returned_corpus_column = returned_corpus_column
        self.k=k
        self.hit_at_k = hit_at_k
        processed_corpus = self._prepare_corpus(corpus_df)      

        super().__init__(corpus=processed_corpus, queries=queries, relevant_docs=relevant_docs, *args, **kwargs)

    def _prepare_corpus(self, documents):
        # Prepares the corpus dictionary and creates a document-paragraph mapping.
        processed_corpus = {}
        doc_paragraph_map = {}
        for idx, row in documents.iterrows():
            processed_corpus[idx] = row['content']
            doc_paragraph_map[idx] = f"d_{row['doc_id']}"
        
        self.doc_paragraph_map = doc_paragraph_map
        return processed_corpus    
    
    def compute_metrics(self, queries_result_list):
        """ Override to compute custom metrics according to document relevance from paragraphs
        and write results including query text and document links to a CSV file. """
        
        query_doc_scores = {}
        csv_data = []

        # Process results for each query separately
        for query_itr, results in enumerate(queries_result_list):
            query_id = self.queries_ids[query_itr]
            query_text = self.queries[query_itr]  # Assuming self.queries stores the actual text of each query
            
            if self.doc_paragraph_map is not None:
                doc_scores = {}            
                for hit in results:
                    paragraph_id = hit['corpus_id']
                    doc_id = self.doc_paragraph_map[paragraph_id]
                    score = hit['score']

                    if doc_id not in doc_scores or score > doc_scores[doc_id]:
                        doc_scores[doc_id] = score

                # Rank documents for this query based on the highest paragraph score
                ranked_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
                query_doc_scores[query_id] = [{'corpus_id': doc_id, 'score': score} for doc_id, score in ranked_docs]
            else:
                # Sort scores
                ranked_docs = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
                query_doc_scores[query_id] = [{'corpus_id': id_score['corpus_id'], 'score': id_score['score']} for id_score in ranked_docs]

            if self.corpus_df is not None:
                # Fetch document links and prepare data for CSV output
                top_k_docs = query_doc_scores[query_id][:self.k]
                top_k_doc_ids = [doc['corpus_id'] for doc in top_k_docs]
                top_k_links = [self.corpus_df[self.corpus_df['doc_id'] == int(doc_id.split('_')[1])][self.returned_corpus_column].values[0] for doc_id in top_k_doc_ids]
                top_k_scores = [doc['score'] for doc in top_k_docs]
                avg_score = sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0
                real_relevant_docs = self.relevant_docs.get(query_id, [])
                real_relevant_links = [self.corpus_df[self.corpus_df['doc_id'] == int(doc_id.split('_')[1])][self.returned_corpus_column].values[0] for doc_id in real_relevant_docs]
                real_relevant_location = []
                
                # Calculating hit@k
                hit_at_k = {}
                for k in self.hit_at_k:
                    top_k_docs = query_doc_scores[query_id][:k]
                    top_k_doc_ids = [doc['corpus_id'] for doc in top_k_docs]
                    returned_relevant_documents = [doc for doc in top_k_doc_ids if doc in real_relevant_docs]
                    hit_at_k[k] = 1 if returned_relevant_documents else 0
                
                for i, x in enumerate(query_doc_scores[query_id]):
                    if x['corpus_id'] in real_relevant_docs:
                        real_relevant_location.append(str(i))
                
                # Padding the results if there are less than self.k docs
                while len(top_k_links) < self.k:
                    top_k_links.append(None)
                
                temp_top_k_links = []
                for score, link in zip(query_doc_scores[query_id], top_k_links):
                    temp_top_k_links.append(link)
                    temp_top_k_links.append(score['score'])
                top_k_links = temp_top_k_links
                    
                # Add query results to the csv data
                csv_data.append([query_text] + top_k_links + ['\n'.join(real_relevant_links), '\n'.join(real_relevant_location), avg_score] + [hit_at_k[k] for k in self.hit_at_k])
          
        if len(csv_data) > 0:
            # Write results to a CSV file
            with open(self.ranking_csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # if self.write_score_per_rank:
                doc_titles = [x for xs in [(f'page {i}', f'score {i}') for i in range(self.k)] for x in xs]
                writer.writerow(['query'] + doc_titles + ['real relevant documents', 'real relevant returned locations', 'average similarity score of top-5 documents'] + [f'hit@{k}' for k in self.hit_at_k])
                writer.writerows(csv_data)
            
        # Return the computed metrics as usual
        return super(CustomInformationRetrievalEvaluator, self).compute_metrics(list(query_doc_scores.values()))

def load_corpus(path, text_column='content'):
    """
    Loads and processes a corpus from a JSON file.

    Reads JSON data into a DataFrame
    Iterates through rows, creating document entries
    Builds a list of document dictionaries with specified fields
    """
    data = pd.read_json(path)
    
    corpus = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        new_item = row[text_column]

        doc = {'doc_id': row['doc_id'], 'paragraph': new_item, 'title': row['title'], 'content': row['content']}
        doc['link'] = row['link']
        corpus.append(doc)

    return pd.DataFrame(corpus)

def create_relevant_docs(df, unique_questions_df, reversed_corpus):
    """
    Creates a dictionary of relevant documents for each unique query.

    This function processes the input DataFrame to identify relevant documents for each query:

    Handles two scenarios based on the presence of an 'doc_id' column in the DataFrame
    For datasets without 'doc_id':
    Matches paragraphs to questions and uses a reversed corpus mapping
    For datasets with 'doc_id':
    Directly uses the 'doc_id' column to identify relevant documents
    Formats query and document IDs consistently
    """
    relevant_docs = {}
    if 'doc_id' not in df.columns:
        # Iterate over each query to find matching paragraphs
        for index, row in unique_questions_df.iterrows():
            question = row['question']
            query_id = f"q_{row['query_id']}"

            # Find document IDs where the paragraph text matches the question exactly
            matching_docs = df[df['question'] == question]['paragraph'].tolist()
            matching_docs_id = []
            for doc in matching_docs:
                matching_docs_id.append(reversed_corpus[doc])
                
            # Update relevant_docs dictionary
            relevant_docs[query_id] = matching_docs_id
    
    else:
        for query_id, row in unique_questions_df.iterrows():
            curr_relevant_docs = df[df['question'] == row['question']]['doc_id'].tolist()
            curr_relevant_docs = [f"d_{int(doc_id)}" for doc_id in curr_relevant_docs]
            relevant_docs[f"q_{query_id}"] = curr_relevant_docs

    return relevant_docs
        
def get_ir_docs_corpus_relevant_docs(df: pd.DataFrame, corpus_path, corpus_text_column):
    ''' Assuming df contains Q&A data with answer_doc_id column attaching each answer to a correspondig
    document from the dump file.
    Assuming corpus data contains doc_id column already.'''

    corpus_df = load_corpus(corpus_path, text_column=corpus_text_column)
    corpus = {f"d_{row['doc_id']}": row['paragraph'] for index, row in corpus_df.iterrows()}
    reversed_corpus = {v:k for k,v in corpus.items()}

    # Initialize dictionaries as before
    queries = {}

    # Process DataFrame as before
    unique_questions_df = df[['question']].drop_duplicates().reset_index(drop=True)

    unique_questions_df['query_id'] = unique_questions_df.index

    queries = {f"q_{row['query_id']}": row['question'] for index, row in unique_questions_df.iterrows()}

    relevant_docs = create_relevant_docs(df, unique_questions_df, reversed_corpus)

    # There shouldn't be questions without answer in the data by this stage
    for qid in unique_questions_df['query_id'].values:
        if f"q_{qid}" not in relevant_docs.keys():
            raise f"Query id {qid} has no answers in relevant docs"

    return queries, corpus, relevant_docs, corpus_df

def load_model_from_artifact(model_name, model_version):
    # Loads a Sentence Transformer model from a local artifact.
    if model_version:
        artifact_dir = os.path.join(os.getcwd(), 'models', f'{model_name}:{model_version}')
    else:    
        artifact_dir = os.path.join(os.getcwd(), 'models', f'{model_name}')
    model = SentenceTransformer(artifact_dir)
    
    return model

def split_train_eval(df, val_size):
    # Splits to train and validation set, maintaining unique questions in only one of the sets.
    unique_questions = df['question'].unique()

    # Split questions into training and validation sets
    train_questions, val_questions = train_test_split(unique_questions, test_size=val_size, random_state=42)

    # Filter the original DataFrame to create training and validation DataFrames
    train_df = df[df['question'].isin(train_questions)]
    val_df = df[df['question'].isin(val_questions)]
    
    return train_df, val_df

def load_val_set(df_val, corpus_text_column, corpus_path):
    queries, corpus, relevant_docs, corpus_df = get_ir_docs_corpus_relevant_docs(df_val,
                                                                      corpus_text_column=corpus_text_column,
                                                                      corpus_path=corpus_path)

    return queries, corpus, relevant_docs, corpus_df

def load_model(model_name, model_version='', use_artifact=False):
    # Loads a Sentence Transformer model either from a local artifact or from a predefined model repository.
    if use_artifact:
        model = load_model_from_artifact(model_name, model_version)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    
    return model

# This sampler assures that questions are unique within the same batch (due to contrastive learning with in-batch negatives).
# We do not want a question appearing twice in the same batch with two different relevant results. If this happens a positive example will be used as a negative example.
class UniqueQuestionSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.questions = [example.texts[0] for example in data_source]
        self.indices = list(range(len(data_source)))

        # Group indices by question
        self.question_to_indices = defaultdict(list)
        for idx, question in enumerate(self.questions):
            self.question_to_indices[question].append(idx)
            
        self.question_groups = list(self.question_to_indices.values())
            
    def _create_batches(self):
        # Convert question groups to a list of deques for easy cycling
        for g in self.question_groups:
            np.random.shuffle(g)
            
        deque_question_groups = [deque(indices) for indices in self.question_groups]
        np.random.shuffle(deque_question_groups)  # Shuffle to ensure variability

        batches = []
        while max([len(g) for g in deque_question_groups]):
            batch = []
            for question_group in deque_question_groups:
                if question_group:
                    batch.append(question_group.popleft())
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []
            # Add any remaining items to the last batch
            if batch:
                batches.append(batch)

        return batches

    def __iter__(self):
        # Dynamically create batches at the beginning of each iteration
        self.batches = self._create_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        # Estimate the number of batches considering the distribution logic
        total_indices = sum(len(indices) for indices in self.question_to_indices.values())
        return (total_indices + self.batch_size - 1) // self.batch_size
