import json
import pandas as pd
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    with open(f'data/ms2/raw/{args.split}.json') as f:
        data = json.load(f)

    paper_set = set()

    for key,value in tqdm(data.items()):
        pmids, titles, abstracts = value['pmid'], value['title'], value['abstract']
        # [t1, t2, t3], [b1,b2,b3], [a1,a2,a3] -> [t1, b1, a1], [t2, b2, a2], [t3, b3, a3]
        for pmid, title, abstract in zip(pmids, titles, abstracts):
            if pmid in paper_set:
                continue
            paper_set.add((pmid, title, abstract))
        
    df = pd.DataFrame(list(paper_set), columns=['pmid', 'title', 'abstract'])

    df.to_csv(f'data/ms2/pm_{args.split}.csv', index=False)
