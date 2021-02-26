import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

# dataset
f = open('myfile.txt', 'r')
data = f.read()
f.close()

df = pd.DataFrame({'文書': [data]})

# exec spacy
nlp = spacy.load('ja_ginza')
docs = [nlp(s) for s in df['文書']]

# 共起語関係取得
df_word_count = pd.concat([utils.get_co_df(d) for d in docs]).reset_index(drop=True)

# 除外する品詞の指定
# ADJ: 形容詞, ADP: 設置詞, ADV: 副詞, AUX: 助動詞, CCONJ: 接続詞, DET: 限定詞, INTJ: 間投詞, NOUN: 名詞, NUM: 数詞,
# PART: 助詞, PRON: 代名詞, PROPN: 固有名詞, PUNCT: 句読点, SCONJ: 連結詞, SYM: シンボル, VERB: 動詞, X: その他
extract_pos = ['ADP', 'INTJ', 'PUNCT', 'SCONJ', 'PART', 'CCONJ', 'AUX', 'DET', 'SCONJ', 'VERB', 'X', 'SYM', 'NUM']

df_ex_word_count = df_word_count[(~df_word_count['word1_pos'].isin(extract_pos)) \
                                 & (~df_word_count['word2_pos'].isin(extract_pos))]

# 共起回数を全文書でまとめておく
df_net = df_ex_word_count.groupby(['word1', 'word2', 'word1_pos', 'word2_pos']).sum() \
    ['count'].sort_values(ascending=False).reset_index()

utils.plot_draw_networkx(df_net)
# utils.plot_draw_networkx(df_net, word='子ども')
