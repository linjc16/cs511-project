from qdrant_client import models, QdrantClient
from load_data import load_csv
import numpy as np
import argparse
import openai
import json
from tqdm import tqdm
from qdrant_client.models import Filter, FieldCondition
from collections import defaultdict

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

def decompe_query(query):
    instruction_prompt = (
        'Decompose the complex query into several parts which will be used for database search, '
        'so that these parts can fully represent the query. '
        'Also, the generated parts do not need to be exactly the same as the original query. '
        'Instead, you can either rephrase the query or extract the most important information. '
        'You should decompose the query in both sentence and keyword level.'
        '\n\n'
        'Query: {query}'
        '\n\n'
        'Now, please decompose the query into several parts and the output should follow the json format as'
        ': sentences: [part1, part2, ...], keywords: [keyword1, keyword2, ...]'
    )

    deco_parts = gpt_chat_35(instruction_prompt, {'query': query})
    # keywords = keywords.split(', ')

    return deco_parts

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
    args = parser.parse_args()
    filepath = args.filepath
    # create_collection()
    # upload_records(filepath)
    
    with open('data/ms2/re_test.json', 'r') as f:
        test_data = json.load(f)
    
    topks = [20, 50, 100]
    
    gt_counts = defaultdict(int)
    pred_count = defaultdict(int)
    for key, value in tqdm(test_data.items()):
        query = value['query']
        ground_truth = value['pmid']
        
        query_vector = openai_client.embeddings.create(
                input=[query],
                model=embedding_model
            ).data[0].embedding

        hits_main = qdrant.search(
            collection_name="clinical_trials_openai",
            query_vector=query_vector,
            limit=max(topks),
        )

        # decompose the query into several keywords
        STOP_SIGNAL = False
        while not STOP_SIGNAL:
            try:
                deco_parts = decompe_query(query)
                deco_parts = json.loads(deco_parts)
                sentences_parts = deco_parts['sentences']
                keywords_parts = deco_parts['keywords']
                STOP_SIGNAL = True
            except Exception as e:
                continue
        

        hits_vecs_sent = []
        for part_query in sentences_parts:
            hits_vec = search_by_parts(part_query, topks)
            hits_vecs_sent.extend(hits_vec)
        
        hits_vecs_word = []
        for part_query in keywords_parts:
            hits_vec = search_by_parts(part_query, topks)
            hits_vecs_word.extend(hits_vec)
        
        

        for topk in topks:
            scores_dict_merge = {}
            scores_main = get_pmid_score_dict(hits_main, topk)
            scores_sent = get_pmid_score_dict(hits_vecs_sent, topk)
            scores_word = get_pmid_score_dict(hits_vecs_word, topk)

            scores_dict_merge = {
                'key': key,
                'scores_main': scores_main,
                'scores_sent': scores_sent,
                'scores_word': scores_word
            }

            # save the dict row by row
            with open(f'data/ms2/results/scores_dict_merge_{topk}.json', 'a') as f:
                f.write(json.dumps(scores_dict_merge) + '\n')

            # merge the scores by 1 * main + 0.2 * sent + 0.2 * word
            scores_merge = defaultdict(float)
            # the scores_merge should have keys from all the scores
            for pmid in set(scores_main.keys()).union(set(scores_sent.keys())).union(set(scores_word.keys())):
                scores_merge[pmid] = scores_main[pmid] + 1.0 / len(hits_vecs_sent) * scores_sent[pmid] + \
                    1.0 / len(hits_vecs_word) * scores_word[pmid]
            
            # sort the scores_merge, from high to low
            scores_merge = dict(sorted(scores_merge.items(), key=lambda x: x[1], reverse=True))
            
            output = list(scores_merge.keys())[:topk]
            # calculate the recall
            recall = len(set(ground_truth).intersection(set(output)))
            gt_counts[topk] += min(len(ground_truth), topk)
            pred_count[topk] += recall

    for topk in topks:
        print(f"Top-{topk}: {pred_count[topk]/gt_counts[topk]}")
