from qdrant_client import models, QdrantClient
from load_data import load_csv
import numpy as np
import argparse
import openai
import json
from tqdm import tqdm
from qdrant_client.models import Filter, FieldCondition
from collections import defaultdict
import glob

import os
import sys
sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_35

tqdm.pandas()
import pdb

with open('openai_api.key', 'r') as f:
    api_key = f.read().strip()

openai_client = openai.Client(
    api_key=api_key
)

embedding_model = "text-embedding-3-small"
qdrant = QdrantClient(path="data/ms2/db/openai")

def create_collection():
    qdrant.recreate_collection(
        collection_name="clinical_trials_openai",
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
        ),
    )

def encode(df):
    texts = []
    for idx, row in tqdm(df.iterrows()):
        fields_to_concat = [row['title'], row['abstract']]
        # Filter out None values
        valid_fields = [field for field in fields_to_concat if field is not None]
        # Concatenate the remaining fields with a space
        concat_str = ' '.join(valid_fields)
        texts.append(concat_str)
    
    # chunk the text, each chunk has 1000 texts
    chunk_size = 1000
    texts_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    embeddings = []
    for chunk in tqdm(texts_chunks):
        result = openai_client.embeddings.create(input=chunk, model=embedding_model)
        embeddings.extend([data.embedding for data in result.data])

    return embeddings

def upload_records(filepath):
    df = load_csv(filepath)
    # # Replace np.nan, np.inf, and -np.inf with None (or any other value you see fit)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)  # First, replace inf and -inf with np.nan
    # df.fillna(value=None, inplace=True)  # Then, replace np.nan with None

    embeddings = encode(df)
    
    row_dict_list = []
    for idx, row in tqdm(df.iterrows()):
        row_dict_list.append(row.to_dict())

    qdrant.upload_points(
        collection_name="clinical_trials_openai",
        points=[
            models.PointStruct(
                id=idx, vector=emb, payload=row_dict
            )
            for idx, (emb, row_dict) in enumerate(zip(embeddings, row_dict_list))
        ],
    )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='data/ms2/pm_test.csv')
    args = parser.parse_args()
    
    filepath = args.filepath
    # create_collection()
    # upload_records(filepath)
    
    with open('data/ms2/re_test.json', 'r') as f:
        test_data = json.load(f)
    
    # topks = [20, 50, 100]
    topks = [100]
    
    gt_counts = defaultdict(int)
    pred_count = defaultdict(int)


    scores_dict_merge_all = {}

    for topk in topks:
        # read the file
        with open(f'data/ms2/results/scores_dict_merge_{topk}.json', 'r') as f:
            scores_ind_dict = {}
            for line in f:
                scores_dict_merge = json.loads(line)
                key = scores_dict_merge['key']
                scores_main = scores_dict_merge['text']['scores_main']
                scores_sent_text = scores_dict_merge['text']['scores_sent']
                scores_word_text = scores_dict_merge['text']['scores_word']
                scores_sent_disease = scores_dict_merge['diseases']['scores_sent']
                scores_word_disease = scores_dict_merge['diseases']['scores_word']
                scores_sent_number = scores_dict_merge['number']['scores_sent']
                scores_sent_treatment = scores_dict_merge['treatments']['scores_sent']
                scores_word_treatment = scores_dict_merge['treatments']['scores_word']

                if key not in scores_ind_dict:
                    scores_ind_dict[key] = {
                        'text': {
                            'scores_main': scores_main,
                            'scores_sent': scores_sent_text,
                            'scores_word': scores_word_text
                        },
                        'diseases': {
                            'scores_sent': scores_sent_disease,
                            'scores_word': scores_word_disease
                        },
                        'number': {
                            'scores_sent': scores_sent_number
                        },
                        'treatments': {
                            'scores_sent': scores_sent_treatment,
                            'scores_word': scores_word_treatment
                        }
                    }
            
            scores_dict_merge_all[topk] = scores_ind_dict

    
    for key, value in tqdm(test_data.items()):
        query = value['query']
        ground_truth = value['pmid']
        
        for topk in topks:
            if key not in scores_dict_merge_all[topk]:
                continue
            scores_dict_merge = scores_dict_merge_all[topk][key]

            scores_main = defaultdict(float)
            scores_main.update(scores_dict_merge['text']['scores_main'])
            scores_sent_text = defaultdict(float)
            scores_sent_text.update(scores_dict_merge['text']['scores_sent'])
            scores_word_text = defaultdict(float)
            scores_word_text.update(scores_dict_merge['text']['scores_word'])

            scores_sent_disease = defaultdict(float)
            scores_sent_disease.update(scores_dict_merge['diseases']['scores_sent'])
            scores_word_disease = defaultdict(float)
            scores_word_disease.update(scores_dict_merge['diseases']['scores_word'])

            scores_sent_number = defaultdict(float)
            scores_sent_number.update(scores_dict_merge['number']['scores_sent'])

            scores_sent_treatment = defaultdict(float)
            scores_sent_treatment.update(scores_dict_merge['treatments']['scores_sent'])
            scores_word_treatment = defaultdict(float)
            scores_word_treatment.update(scores_dict_merge['treatments']['scores_word'])
            
            # merge the scores by 1 * main + 0.2 * sent + 0.2 * word
            scores_merge = defaultdict(float)

            union_set = set(scores_main.keys()).union(set(scores_sent_text.keys())).union(set(scores_word_text.keys())) \
                .union(set(scores_sent_disease.keys())).union(set(scores_word_disease.keys())) \
                .union(set(scores_sent_number.keys())).union(set(scores_sent_treatment.keys())).union(set(scores_word_treatment.keys()))
            
            
            for pmid in union_set:
                scores_merge[pmid] = scores_main[pmid] + 0.2 * (
                    scores_sent_text[pmid] + scores_word_text[pmid] + scores_sent_disease[pmid] + 
                    scores_word_disease[pmid] + scores_sent_number[pmid] + scores_sent_treatment[pmid] + 
                    scores_word_treatment[pmid]
                )

            # sort the scores_merge, from high to low
            scores_merge = dict(sorted(scores_merge.items(), key=lambda x: x[1], reverse=True))

            output = list(scores_merge.keys())[:topk]
            # calculate the recall
            recall = len(set(ground_truth).intersection(set(output)))
            gt_counts[topk] += min(len(ground_truth), topk)
            pred_count[topk] += recall

    for topk in topks:
        print(f"Top-{topk}: {pred_count[topk]/gt_counts[topk]}")
