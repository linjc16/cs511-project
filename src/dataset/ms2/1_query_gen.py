import os
import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm
import pdb

sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_35


if __name__ == '__main__':
    
    instruction_prompt = (
        "Based on the detailed background information and critical insights provided in a review for a collection of scholarly papers, construct a user search query. "
        "The generated query should contain all the information from the background and review provided. "
        "The query should be able to retrieve the collection of scholarly papers. "
        "The query should be in the form of a user search query. A user search query is a query that a user would type into a search engine to retrieve the desired information. "
        "Make the query complex enough and maybe a very long paragraph so that I can construct a extremely challenging dataset."
        "\n\n"
        "Background Information: {background}"
        "\n"
        "Review: {review}"
        "\n\n"
        "Please generate a user search query based on the background information and review provided."
    )

    with open('data/ms2/raw/test.json') as f:
        data = json.load(f)

    generated_data = {}

    i = 0
    for key, value in tqdm(data.items()):
        background, review = value['background'], value['target']

        try:
            gen_query = gpt_chat_35(instruction_prompt, {'background': background, 'review': review})
        except:
            gen_query = ""
        
        generated_data[key] = {
            "pmid": value['pmid'],
            "query": gen_query
        }
        
        if i % 10 == 0:
            with open('data/ms2/re_test.json', 'w') as f:
                json.dump(generated_data, f, indent=4)
        
        i += 1
    
    with open('data/ms2/re_test.json', 'w') as f:
        json.dump(generated_data, f, indent=4)