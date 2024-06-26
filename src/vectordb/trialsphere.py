from qdrant_client import models, QdrantClient
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
import pandas as pd

def load_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.astype(str)
    df = df.where(pd.notnull(df), None)
    df = df.map(lambda x: None if pd.isna(x) else x)
    # pd.set_option('display.max_columns', None)
    # print(df.head)
    print("read completed")
    return df

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



def get_pmid_score_dict(hits, topk):
    pmid_score_dict = defaultdict(float)
    hits_subset = hits[:topk]
    for hit in hits_subset:
        pmid = hit.payload['pmid']
        score = hit.score
        pmid_score_dict[pmid] += score

    return pmid_score_dict


def search_by_parts(part_query, topks):
    query_vector = openai_client.embeddings.create(
            input=[part_query],
            model=embedding_model
        ).data[0].embedding
    
    # first search by the vector
    hits_vec = qdrant.search(
        collection_name="clinical_trials_openai",
        query_vector=query_vector,
        limit=max(topks)  # Return 5 closest points
    )

    # # then search for keywords
    # hits = qdrant.search(
    #     collection_name="clinical_trials_openai",
    #     query_vector=query_vector,
    #     query_filter=Filter(
    #         must=[  # These conditions are required for search results
    #             FieldCondition(
    #                 key='title',  # Condition based on values of `rand_number` field.
    #                 match=models.MatchText(text="necrotizing enterocolitis"),
    #             )
    #         ]
    #     ),
    #     limit=max(topks)  # Return 5 closest points
    # )

    return hits_vec


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='data/ms2/pm_test.csv')
    # parser.add_argument('--load_scores', action='store_true')
    args = parser.parse_args()
    args.load_scores = False
    
    filepath = args.filepath
    # create_collection()
    # upload_records(filepath)
    
    with open('data/ms2/re_test.json', 'r') as f:
        test_data = json.load(f)
    
    topks = [20, 50, 100]
    
    gt_counts = defaultdict(int)
    pred_count = defaultdict(int)

    # load the decomposed queries
    decomposed_queries = {}
    filepaths = glob.glob('data/ms2/results/raw/gen_parts/decomposed_query_*.json')
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            decomposed_queries.update(json.load(f))

    # pdb.set_trace()
    if args.load_scores:
        scores_dict_merge_all = {}

        for topk in topks:
            # read the file
            with open(f'data/ms2/results/scores_dict_merge_{topk}.json', 'r') as f:
                scores_ind_dict = {}
                for line in f:
                    scores_dict_merge = json.loads(line)
                    key = scores_dict_merge['key']
                    scores_main = scores_dict_merge['scores_main']
                    scores_sent = scores_dict_merge['scores_sent']
                    scores_word = scores_dict_merge['scores_word']

                    if key not in scores_ind_dict:
                        scores_ind_dict[key] = {
                            'scores_main': scores_main,
                            'scores_sent': scores_sent,
                            'scores_word': scores_word
                        }
                
                scores_dict_merge_all[topk] = scores_ind_dict


    for key, value in tqdm(test_data.items()):
        query = value['query']
        ground_truth = value['pmid']
        
        if not args.load_scores:
            query_vector = openai_client.embeddings.create(
                    input=[query],
                    model=embedding_model
                ).data[0].embedding

            hits_main = qdrant.search(
                collection_name="clinical_trials_openai",
                query_vector=query_vector,
                limit=max(topks),
            )


            sentences_parts = decomposed_queries[key]['sentences_parts']
            keywords_parts = decomposed_queries[key]['keywords_parts']

            hits_vecs_sent = []
            for part_query in sentences_parts:
                hits_vec = search_by_parts(part_query, topks)
                hits_vecs_sent.extend(hits_vec)
        
            hits_vecs_word = []
            for part_query in keywords_parts:
                hits_vec = search_by_parts(part_query, topks)
                hits_vecs_word.extend(hits_vec)
        
        
        for topk in topks:
            # if scores_dict_merge_all empyt[topk]
            if not args.load_scores:
                scores_dict_merge = {}
                scores_main = get_pmid_score_dict(hits_main, topk)
                scores_sent = get_pmid_score_dict(hits_vecs_sent, topk)
                # scores_word = get_pmid_score_dict(hits_vecs_word, topk)
                scores_word = defaultdict(float)

                scores_dict_merge = {
                    'key': key,
                    'scores_main': scores_main,
                    'scores_sent': scores_sent,
                    'scores_word': scores_word
                }

                # save the dict row by row
                with open(f'data/ms2/results/scores_dict_merge_{topk}.json', 'a') as f:
                    f.write(json.dumps(scores_dict_merge) + '\n')

            else:
                scores_dict_merge = scores_dict_merge_all[topk][key]
                scores_main = defaultdict(float)
                scores_main.update(scores_dict_merge['scores_main'])
                scores_sent = defaultdict(float)
                scores_sent.update(scores_dict_merge['scores_sent'])
                scores_word = scores_dict_merge['scores_word']
                scores_word = defaultdict(float)
            
            scores_merge = defaultdict(float)
            
            # the scores_merge should have keys from all the scores
            for pmid in scores_main.keys():
                scores_merge[pmid] = scores_main[pmid]
            
            # sort the scores_merge, from high to low
            scores_merge = dict(sorted(scores_merge.items(), key=lambda x: x[1], reverse=True))

            output = list(scores_merge.keys())[:topk]
            # calculate the recall
            recall = len(set(ground_truth).intersection(set(output)))
            gt_counts[topk] += min(len(ground_truth), topk)
            pred_count[topk] += recall

    for topk in topks:
        print(f"Top-{topk}: {pred_count[topk]/gt_counts[topk]}")
