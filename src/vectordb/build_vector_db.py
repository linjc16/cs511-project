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
from load_data import load_csv
import numpy as np


encoder = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
qdrant = QdrantClient(path="data/ms2/db")

def create_collection():
    qdrant.recreate_collection(
        collection_name="clinical_trials",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

def create_multivector_collection():
    qdrant.recreate_collection(
        collection_name='clinical_trials_mv',
        vectors_config={
            "title": models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
            "abstract": models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        }
    )

def encode(row):
    # fields_to_concat = [row['Public Title'], row['Scientific Title'], row['Brief Summary']]
    fields_to_concat = [row['title'], row['abstract']]
    # Filter out None values
    valid_fields = [field for field in fields_to_concat if field is not None]
    # Concatenate the remaining fields with a space
    concat_str = ' '.join(valid_fields)
    return encoder.encode(concat_str).tolist()

def upload_records(filepath):
    df = load_csv(filepath)
    # # Replace np.nan, np.inf, and -np.inf with None (or any other value you see fit)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)  # First, replace inf and -inf with np.nan
    # df.fillna(value=None, inplace=True)  # Then, replace np.nan with None

    qdrant.upload_points(
        collection_name="clinical_trials",
        points=[
            models.PointStruct(
                id=idx, vector=encode(row), payload=row.to_dict()
            )
            for idx, row in df.iterrows()
        ],
    )

def encode_multivector(row):
    title, abstract = row['title'], row['abstract']
    encoded_title = encoder.encode(title).tolist()
    encoded_abstract = encoder.encode(abstract).tolist()
    return {"title": encoded_title, "abstract": encoded_abstract}


def upload_multivector_records(filepath):
    df = load_csv(filepath)
    qdrant.upsert(
        collection_name="clinical_trials_mv",
        points=[
            models.PointStruct(
                id=idx,
                vector=encode_multivector(row),
                payload=row.to_dict()
            )
            for idx, row in df.iterrows()
        ]
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

    # multi vector mode
    create_multivector_collection()
    upload_multivector_records(filepath)

    query = 'Give me the records about methylphenidate lab'
    hits = qdrant.search(
        collection_name="clinical_trials_mv",
        query_vector=models.NamedVector(
            name="abstract",
            vector=encoder.encode(query).tolist(),
        ),
        limit=10,
        with_vectors=False,
        with_payload=True,
    )

    for hit in hits:
        print(hit.payload, "score:", hit.score)


