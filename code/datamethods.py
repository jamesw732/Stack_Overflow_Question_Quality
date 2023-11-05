import json
from stackapi import StackAPI


def get_raw_data(filename, startdate):
    """ Gets first 500 questions from startdate and the next 10 days and dumps them into a JSON.
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