import sqlite3
import csv

def create_trial_table(cursor):
    """Create the trial table if not exist, if exists, first drop and then recreate"""

    cursor.execute("DROP TABLE IF EXISTS trial")
    sql = '''
        CREATE TABLE trial(
            pmid INTEGER PRIMARY KEY, 
            title TEXT, 
            abstract TEXT
        )'''
    cursor.execute(sql)

    print('successfully created trial table')


def load_data_from(file_path, cursor):
    """Load data from a csv file into the trial table"""

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        insert_query = 'INSERT INTO trial (pmid, title, abstract) VALUES (?, ?, ?)'

        count = 0
        valid_count = 0
        for row in csvreader:
            if row['PMID'] and row['Title'] and row['Abstract']:
                cursor.execute(insert_query, (row['PMID'], row['Title'], row['Abstract']))
                valid_count += 1
            count += 1

        print(f'load completed, total records: {count}, valid records: {valid_count}')


def retrieve_records(cursor):
    """Return the number of records contaning blood in their abstracts"""

    query = "SELECT COUNT(*) FROM trial WHERE abstract LIKE '%blood%'"
    cursor.execute(query)
    result = cursor.fetchone()
    print(f'there are {result} records having \"blood\" in their abstracts')


if __name__=='__main__':
    connection = sqlite3.connect("clinicals.db")
    cursor = connection.cursor()

    # create table
    # ONLY execute this IF
    # this is your first time running the program 
    # OR
    # you changed the schema of the trial table
    create_trial_table(cursor)

    # load data into the trial table
    load_data_from('./output_test1.csv', cursor)

    # commit the changes if any
    connection.commit()

    # run a simple test
    retrieve_records(cursor)

    # always close the connection
    connection.close()
