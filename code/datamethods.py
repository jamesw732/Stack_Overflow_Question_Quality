import json
from stackapi import StackAPI
import re
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
import os
import time


datadir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../data'))


def get_raw_data(filename, startdate):
    """Gets first 500-ish questions from startdate until keyboard interrupt and dumps them into a JSON.

    filename: str, json filename. Careful here, don't overwrite any data.
    startdate: int, seconds since 1/1/1970.
    """
    site = StackAPI('stackoverflow', key='uRxCi2XFwSnvGxFe38PEFA((')
    posts = []
    i = 0
    try:
        while True:    
            posts += site.fetch('questions', filter='withbody', 
                            fromdate=startdate + 84600 * i, 
                            todate=startdate + 10000 + 84600 * i)['items']
            i += 1
            print(i)
            time.sleep(0.2)
    finally:
        posts_by_id = {int(item['question_id']): item for item in posts}

        with open(filename, 'w') as f:
            json.dump(posts_by_id, f, indent=4)

def str_len_no_whitespace(s):
    """Returns the length of given string without considering whitespace"""
    return len(s.translate({ord(c): None for c in string.whitespace}))


def process_raw(infile, outfile):
    """Processes raw json into a processed csv.

    infile: str, input json filename
    outfile: str, output csv filename"""
    with open(infile) as f:
        indict = json.load(f)
    indata = pd.DataFrame(indict.values())
    outdata = pd.DataFrame()
    outdata['score'] = indata['score']
    # Metadata Features (May want to exclude from logistic for interpretability):
    outdata['is_answered'] = indata['is_answered']
    outdata['owner_rep'] = [row.get('reputation', 0) for row in indata['owner']]
    outdata['owner_accept_rate'] = [row.get('accept_rate', 0) for row in indata['owner']]
    # Primary Data Features:
    html_bodies = indata['body']
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    # Use clean_bodies for any natural language processing
    clean_bodies = [re.sub(CLEANR, '', html_body) for html_body in html_bodies]

    outdata['title_len'] = [len(title) for title in indata['title']]
    outdata['body_length'] = [str_len_no_whitespace(body) for body in clean_bodies]
    codeblocks = [re.findall(r'<code>\s*(.*?)\s*</code>', body) for body in html_bodies]
    outdata['num_codeblocks'] = [len(l) for l in codeblocks]
    outdata['len_codeblocks'] = [sum([str_len_no_whitespace(block) for block in codeblock]) for codeblock in codeblocks]
    outdata['code/text'] = outdata['len_codeblocks'] / outdata['body_length']
    outdata['num_tags'] = [len(tags) for tags in indata['tags']]
    outdata = outdata[outdata['score'] != 0]

    outdata.to_csv(outfile, index=False)

def split(filename):
    with open(filename) as f:
        df = pd.read_csv(f)
    train, test = train_test_split(df)
    train.to_csv(os.path.join(datadir, 'train.csv'), index=False)
    test.to_csv(os.path.join(datadir, 'test.csv'), index=False)

if __name__ == '__main__':
    # get_raw_data('data/raw_data2.json', 1609459200)
    process_raw('data/raw_data2.json', 'data/processed_data.csv')
    split('data/processed_data.csv')