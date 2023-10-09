"""Only meant to be run once. Takes the first 500 questions from each day
between oct 1 2020 and oct 10 2020 and puts them in raw_data.json"""
import json
from stackapi import StackAPI

site = StackAPI('stackoverflow', key='r)LaTmhkMjEN8CXvN5JLng((')
posts = []
for i in range(10):
    posts += site.fetch('questions', filter='withbody', 
                       fromdate=1601510400 + 84600 * i, 
                       todate=1601510400 + 84600 * i  + 42300)['items']
posts_by_id = {int(item['question_id']): item for item in posts}

with open('../data/raw_data.json', 'w') as f:
    json.dump(posts_by_id, f, indent=4)