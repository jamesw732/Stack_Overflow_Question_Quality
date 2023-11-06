import json
from stackapi import StackAPI
import re
import pandas as pd
import numpy as np
import string


def get_raw_data(filename, startdate):
    """Gets first 500 questions from startdate and the next 10 days and dumps them into a JSON.

    filename: str, json filename. Careful here, don't overwrite any data.
    startdate: int, seconds since 1/1/1970.
    """
    site = StackAPI('stackoverflow', key='r)LaTmhkMjEN8CXvN5JLng((')
    posts = []
    for i in range(10):
        posts += site.fetch('questions', filter='withbody', 
                        fromdate=startdate + 84600 * i, 
                        todate=startdate + 42300 + 84600 * i)['items']
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
    outdata['view_count'] = indata['view_count']
    outdata['answer_count'] = indata['answer_count']
    outdata['owner_rep'] = [row.get('reputation', 0) for row in indata['owner']]
    outdata['owner_accept_rate'] = [row.get('accept_rate', 0) for row in indata['owner']]
    outdata['is_closed'] = np.zeros(shape=len(indata), dtype='i1')
    outdata['is_closed'][~np.isnan(indata['closed_date'])] = 1
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

    outdata.to_csv(outfile)