import sqlite3
import csv
import json
from collections import defaultdict
from tqdm import tqdm
import glob
import pdb

def create_trial_table(cursor):
    """Create the trial table if not exist, if exists, first drop and then recreate"""

    cursor.execute("DROP TABLE IF EXISTS trial")
    sql = '''
        CREATE TABLE trial(
            pmid INTEGER PRIMARY KEY, 
            title TEXT, 
            abstract TEXT,
            date TEXT
        )'''
    cursor.execute(sql)

    print('successfully created trial table')


def load_data_from(file_path, cursor):
    """Load data from a csv file into the trial table"""

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        insert_query = 'INSERT INTO trial (pmid, title, abstract, date) VALUES (?, ?, ?, ?)'

        count = 0
        valid_count = 0
        for row in csvreader:
            if row['pmid'] and row['title'] and row['abstract'] and row['date']:
                cursor.execute(insert_query, (row['pmid'], row['title'], row['abstract'], row['date']))
                valid_count += 1
            count += 1

        print(f'load completed, total records: {count}, valid records: {valid_count}')


def retrieve_records(cursor, keywords):
    """Return the records containing any of the keywords in their titles or abstracts"""
    
    patterns = ['%' + keyword + '%' for keyword in keywords]
    
    query = "SELECT pmid, title, abstract, date FROM trial WHERE "
    conditions = []
    for pattern in patterns:
        conditions.append("(title LIKE ? OR abstract LIKE ?)")
    query += " OR ".join(conditions)
    
    args = [pattern for pattern in patterns for _ in (1, 2)]
    cursor.execute(query, args)
    results = cursor.fetchall()
    
    # print(f'Found {len(results)} records containing any of the keywords: {", ".join(keywords)}')
    return results
    

if __name__=='__main__':
    connection = sqlite3.connect("clinicals_sql.db")
    cursor = connection.cursor()

    # create table
    # ONLY execute this IF
    # this is your first time running the program 
    # OR
    # you changed the schema of the trial table
    create_trial_table(cursor)

    # load data into the trial table
    load_data_from('data/ms2/pm_test.csv', cursor)

    # commit the changes if any
    connection.commit()


    with open('data/ms2/re_test.json', 'r') as f:
        test_data = json.load(f)
    
    topks = [20, 50, 100]


    gt_counts = defaultdict(int)
    pred_count = defaultdict(int)

    decomposed_queries_text = {}
    keyword_filepaths = glob.glob('data/ms2/results/raw/gen_parts/text/decomposed_query*.json')
    for filepath in keyword_filepaths:
        with open(filepath, 'r') as f:
            decomposed_queries_text.update(json.load(f))
    
    for key, value in tqdm(test_data.items()):
        query = value['query']
        ground_truth = value['pmid']

        keywords = decomposed_queries_text[key]['keywords_parts']

        # run a simple test
        res = retrieve_records(cursor, keywords)
        for topk in topks:
            output = [hit[0] for hit in res[:topk]]
            # calculate the recall
            recall = len(set(ground_truth).intersection(set(output)))
            gt_counts[topk] += min(len(ground_truth), topk)
            pred_count[topk] += recall
    
    for topk in topks:
        print(f"Top-{topk}: {pred_count[topk]/gt_counts[topk]}")

    # always close the connection
    connection.close()
