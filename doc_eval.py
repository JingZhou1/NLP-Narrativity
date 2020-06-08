import pandas as pd

######################
#  Hyper-parameters  #
######################

# how many grammar errors in a sentence can be considered as too many?
numOfErr = 3

# document score is the weighted sum of the scores in the next section
# the weights are hyper-parameters now and can be trained later if we know which documents are good/bad
weights = [0.1, 0.2, 0.3, 0.4] # weights for (i) standard deviation, (ii) mean, (iii) domain specific score, and (iv) more important specific score


################################################
#  Definition of Each Score for Grammar_Errors #
################################################

df = pd.read_csv('grammar_errors.csv')

# obtain the groupby objects for statistics
df_score_groupby_doc = df.groupby(['doc_id'])['grammar_error_score']
df_count_groupby_doc = df.groupby(['doc_id'])['grammar_error_count']

# statistics of the scores of the sentences
df_score_stat = df_score_groupby_doc.agg(['mean', 'count', 'std'])

# 1. Standard deviation of sentence grammar error score
df_score_std = df_score_stat['std']

# 2. Average sentence grammar error score
df_score_mean = df_score_stat['mean']

# 3. Percentage of sentences with grammar errors
df_score_eq0 = df_score_groupby_doc.agg(lambda x: x.astype(float).eq(0).sum())
df_per_sent_w_grammar_err = (df_score_stat['count'] - df_score_eq0) / df_score_stat['count']

# 4. Percentage of sentences with at least 3 grammar errors
df_count_geq3 = df_count_groupby_doc.agg(lambda x: x.astype(float).ge(numOfErr).sum())
df_per_sent_w_geq3_grammar_err = df_count_geq3 / df_score_stat['count']

df_Grammar_Errors = df_score_std * weights[0] + df_score_mean * weights[1] + df_per_sent_w_grammar_err * weights[2] + df_per_sent_w_geq3_grammar_err * weights[3]
df_final = pd.DataFrame({'DocumentID': df_Grammar_Errors.index, 'Grammar_Errors': df_Grammar_Errors.values})



##################################################
#  Definition of Each Score for Sentences_Scored #
##################################################

df = pd.read_csv('sentences_scored.csv')

# obtain the groupby objects for statistics
df_score_groupby_doc = df.groupby(['doc_id'])['parse_score']

# statistics of the scores of the sentences
df_score_stat = df_score_groupby_doc.agg(['mean', 'std'])

# 1. Standard deviation of sentence grammar error score
df_score_std = df_score_stat['std']

# 2. Average sentence grammar error score
df_score_mean = df_score_stat['mean']

df_Sentences_Scored = df_score_std * weights[0] + df_score_mean * weights[1]
df_final['Sentences_Scored'] = df_Sentences_Scored.values



#####################################################
#  Definition of Each Score for Word_Sentence_Score #
#####################################################

df = pd.read_csv('word_sentence_score.csv')

# obtain the groupby objects for statistics
df_score_groupby_doc = df.groupby(['doc_id'])['word_score']

# statistics of the scores of the sentences
df_score_stat = df_score_groupby_doc.agg(['mean', 'std'])

# 1. Standard deviation of sentence grammar error score
df_score_std = df_score_stat['std']

# 2. Average sentence grammar error score
df_score_mean = df_score_stat['mean']

df_Word_Sentence_Score = df_score_std * weights[0] + df_score_mean * weights[1]
df_final['Word_Sentence_Score'] = df_Word_Sentence_Score.values



##############################
#  Save the Final DataFrame  #
##############################

df_final.to_csv('results.csv', index=False)