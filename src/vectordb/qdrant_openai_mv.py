# there are currently several options for building the vector db
# 1) Use multiple vector representations for different columns
# 2) Concatenate the columns we want to embed and use a single vector representation
# 3) Embed the columns individually and somehow combine the embedding vectors into one
# For method 1), we need to consider which column to query against when doing queries
# an idea could be using novel methods to generate plans querying specific columns from
# natural language
# For method 2 and 3, it is much easier to do a query since we are only querying against
# one vector

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm
import pdb
import json

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

qdrant = QdrantClient(path="data/ms2/db/mv_openai")


def create_multivector_collection():
    qdrant.recreate_collection(
        collection_name='clinical_trials_mv',
        vectors_config={
            "title": models.VectorParams(
                size=1536, 
                distance=models.Distance.COSINE,
            ),
            "abstract": models.VectorParams(
                size=1536, 
                distance=models.Distance.COSINE,
            ),
        }
    )


# def encode_multivector(row):
#     title, abstract = row['title'], row['abstract']
#     encoded_title = encoder.encode(title).tolist()
#     encoded_abstract = encoder.encode(abstract).tolist()
#     return {"title": encoded_title, "abstract": encoded_abstract}

# def upload_multivector_records(filepath):
#     df = load_csv(filepath)
#     qdrant.upsert(
#         collection_name="clinical_trials_mv",
#         points=[
#             models.PointStruct(
#                 id=idx,
#                 vector=encode_multivector(row),
#                 payload=row.to_dict()
#             )
#             for idx, row in df.iterrows()
#         ]
#     )

def encode_multivector(df):
    title_embeddings = []
    abstract_embeddings = []

    title_texts = [row['title'] for idx, row in df.iterrows()]
    chunk_size = 1000
    title_chunks = [title_texts[i:i + chunk_size] for i in range(0, len(title_texts), chunk_size)]
    for chunk in tqdm(title_chunks):
        result = openai_client.embeddings.create(input=chunk, model=embedding_model)
        title_embeddings.extend([data.embedding for data in result.data])
    
    abstract_texts = [row['abstract'] for idx, row in df.iterrows()]
    chunk_size = 1000
    abstract_chunks = [abstract_texts[i:i + chunk_size] for i in range(0, len(abstract_texts), chunk_size)]
    for chunk in tqdm(abstract_chunks):
        result = openai_client.embeddings.create(input=chunk, model=embedding_model)
        abstract_embeddings.extend([data.embedding for data in result.data])
    
    embeddings = []
    for i in range(len(df)):
        title_embedding = title_embeddings[i] if i < len(title_embeddings) else []
        abstract_embedding = abstract_embeddings[i] if i < len(abstract_embeddings) else []
        embeddings.append({"title": title_embedding, "abstract": abstract_embedding})

    return embeddings

def upload_multivector_records(filepath):
    df = load_csv(filepath)

    embeddings = encode_multivector(df)

    row_dict_list = []
    for idx, row in tqdm(df.iterrows()):
        row_dict_list.append(row.to_dict())
    
    qdrant.upsert(
        collection_name="clinical_trials_mv",
        points=[
            models.PointStruct(
                id=idx,
                vector=emb,
                payload=row_dict
            )
            for idx, (emb, row_dict) in enumerate(zip(embeddings, row_dict_list))
        ],
    )


if __name__=='__main__':
    filepath = 'data/ms2/pm_test.csv'
    # concat mode
    # create_collection()
    # upload_records(filepath)
    
    # # query = "What is the impact of methylphenidate on academic productivity and accuracy in children with ADHD? Are there any mediating or moderating effects of symptom improvements, demographic factors, design variables, or disorder-related variables? What are the findings of previous reviews on stimulant-related academic improvements? Are there any recent studies that suggest outcome-domain-specific medication effects? How do these effects compare in terms of productivity and accuracy for math, reading, and spelling? What is the magnitude of academic improvements compared to symptom improvements? Are there any qualitative changes observed in math?"
    # query = 'methylphenidate'
    # hits = qdrant.search(
    #     collection_name="clinical_trials",
    #     query_vector=encoder.encode(query).tolist(),
    #     limit=30,
    # )
    # for hit in hits:
    #     print(hit.payload, "score:", hit.score)

    # # multi vector mode
    # create_multivector_collection()
    # upload_multivector_records(filepath)

    # query = 'Give me the records about methylphenidate lab'
    # hits = qdrant.search(
    #     collection_name="clinical_trials_mv",
    #     query_vector=models.NamedVector(
    #         name="abstract",
    #         vector=encoder.encode(query).tolist(),
    #     ),
    #     limit=10,
    #     with_vectors=False,
    #     with_payload=True,
    # )


    # for hit in hits:
    #     print(hit.payload, "score:", hit.score)


    
    with open('data/ms2/re_test.json', 'r') as f:
        test_data = json.load(f)
    
    topks = [20, 50, 100]
    
    gt_counts = {}
    pred_count = {}
    i = 0
    for key, value in tqdm(test_data.items()):
        if i > 121:
            break
        i += 1

        query = value['query']
        ground_truth = value['pmid']
        
        hits = qdrant.search(
            collection_name="clinical_trials_mv",
            query_vector=openai_client.embeddings.create(
                input=[query],
                model=embedding_model
            ).data[0].embedding,
            limit=max(topks),
        )
        for topk in topks:
            if topk not in gt_counts:
                gt_counts[topk] = 0
                pred_count[topk] = 0
            output = [hit.payload['pmid'] for hit in hits[:topk]]
            # calculate the recall
            recall = len(set(ground_truth).intersection(set(output)))
            gt_counts[topk] += min(len(ground_truth), topk)
            pred_count[topk] += recall

    for topk in topks:
        print(f"Top-{topk}: {pred_count[topk]/gt_counts[topk]}")