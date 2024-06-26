from qdrant_client import models, QdrantClient
import numpy as np
import argparse
import openai
import json
from tqdm import tqdm
from qdrant_client.models import Filter, FieldCondition
from collections import defaultdict
import multiprocessing

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

def decomp_query_text(query):
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

    return deco_parts


def decomp_query_trea_dise(query):
    instruction_prompt = (
        'Given the query, extract the essential information related to diseases and treatments at both the keyword and sentence levels.'
        '\n\n'
        'Query: {query}'
        '\n\n'
        'Now, please decompose the query into several parts and the output should follow the json format as: '
        '```'
        'diseases: sentences: [part1, part2, ...], keywords: [keyword1, keyword2, ...]\n'
        'treatments: sentences: [part1, part2, ...], keywords: [keyword1, keyword2, ...]'
        '```'
    )

    deco_parts = gpt_chat_35(instruction_prompt, {'query': query})

    return deco_parts

def decomp_query_number(query):

    instruction_prompt = (
        'Given the query, extract the essential information related to exact numbers like at the sentence level. If no exact numbers are present, please return an empty list. For example, if the query is "What is the average age of patients in the study?", the output should be [] since the sentence does not contain any numbers. '
        'If the query is "The average age of patients in the study is 50 years old.", the output should be ["The average age of patients in the study is 50 years old."].'
        '\n\n'
        'Query: {query}'
        '\n\n'
        'Now, please decompose the query into several parts with numbers and the output should follow the json format as: '
        '```'
        'sentences: [part1, part2, ...]'
        '```'
    )

    deco_parts = gpt_chat_35(instruction_prompt, {'query': query})

    return deco_parts

def worker(input):
    process_id, test_data = input

    output_dict = {}
    i = 0
    for key, value in tqdm(test_data.items()):
        query = value['query']

        # decompose the query into several keywords
        STOP_SIGNAL = False
        while not STOP_SIGNAL:
            try:
                if args.mode == 'text':
                    deco_parts = decomp_query_text(query)
                    deco_parts = json.loads(deco_parts)
                    sentences_parts = deco_parts['sentences']
                    keywords_parts = deco_parts['keywords']
                    STOP_SIGNAL = True
                    output_dict[key] = {
                        'sentences_parts': sentences_parts,
                        'keywords_parts': keywords_parts
                    }
                elif args.mode == 'trea_dise':
                    deco_parts = decomp_query_trea_dise(query)
                    deco_parts = json.loads(deco_parts)
                    output_dict[key] = deco_parts
                    STOP_SIGNAL = True
                elif args.mode == 'number':
                    deco_parts = decomp_query_number(query)
                    deco_parts = json.loads(deco_parts)
                    output_dict[key] = deco_parts
                    STOP_SIGNAL = True
                else:
                    raise ValueError('Invalid mode')

            
            except Exception as e:
                continue
        

        if i % 100 == 0:
            with open(f'data/ms2/results/raw/gen_parts/{args.mode}/decomposed_query_{process_id}.json', 'w') as f:
                json.dump(output_dict, f, indent=4)
        
        i += 1
    
    with open(f'data/ms2/results/raw/gen_parts/{args.mode}/decomposed_query_{process_id}.json', 'w') as f:
        json.dump(output_dict, f, indent=4)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='data/ms2/pm_test.csv')
    parser.add_argument('--mode', type=str, default='text')
    args = parser.parse_args()
    filepath = args.filepath
    # create_collection()
    # upload_records(filepath)
    save_dir = f'data/ms2/results/raw/gen_parts/{args.mode}/'
    os.makedirs(save_dir, exist_ok=True)
    
    with open('data/ms2/re_test.json', 'r') as f:
        test_data = json.load(f)

    num_processes = 10

    # split test_data into num_processes parts
    test_data_parts = []
    num_data = len(test_data)
    num_data_per_part = num_data // num_processes
    for i in range(num_processes):
        if i == num_processes - 1:
            test_data_parts.append({k: v for k, v in test_data.items() if int(k) >= i * num_data_per_part})
        else:
            test_data_parts.append({k: v for k, v in test_data.items() if i * num_data_per_part <= int(k) < (i + 1) * num_data_per_part})
    
    # add procee_id to each part
    test_data_parts = [(i, test_data_part) for i, test_data_part in enumerate(test_data_parts)]
    
    pool = multiprocessing.Pool(num_processes)
    
    # pool.map(worker, test_data_parts)
    worker(test_data_parts[0])