from sklearn.model_selection import train_test_split
import os
import pathlib
import pandas as pd

datadir = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../../data'))

df = pd.read_csv(os.path.join(datadir, 'processed_data.csv'))
df = df[df['score'] != 0]
good_score = df['score'] > 0
good_score.name = 'good_score'
answered = df['is_answered']
metadata = ['score', 'owner_rep', 'owner_accept_rate', 'is_closed', 'view_count', 'answer_count', 'is_answered']
df_input = df[df.columns.drop(metadata)]

X_train, X_test, y_train, y_test = train_test_split(df_input, good_score)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
train.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(),'score_train.csv'), index=False)
test.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(),'score_test.csv'), index=False)

X_train, X_test, y_train, y_test = train_test_split(df_input, answered)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
train.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(), 'answered_train.csv'), index=False)
test.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(), 'answered_test.csv'), index=False)