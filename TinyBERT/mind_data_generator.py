import os

from transformer import BertTokenizer

MAX_TITLE_LEN = 10
MAX_USER_HIS = 10

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

mode = TRAIN

data_dir = 'glue_data/MIND'
news_data = os.path.join(data_dir, os.path.join(mode, 'news.tsv'))
interaction_data = os.path.join(data_dir, os.path.join(mode, 'behaviors.tsv'))

if mode is TRAIN:
    main_data = os.path.join(data_dir, 'train.tsv')
    sub_data = os.path.join(data_dir, 'dev.tsv')
    sub_ratio = 0.1
else:
    main_data = os.path.join(data_dir, 'test.tsv')
    sub_data = os.path.join(data_dir, '_fake.tsv')
    sub_ratio = 0

tokenizer = BertTokenizer.from_pretrained('fake_bert_model')

news_dict = {}
charset = set()

with open(news_data, 'r') as f:
    news_list = f.readlines()

with open(interaction_data, 'r') as f:
    interaction_list = f.readlines()

for news in news_list:
    news_features = news.split('\t')
    id_ = news_features[0]
    title = news_features[3]
    limit_title = ''.join(tokenizer.tokenize(title)[:MAX_TITLE_LEN]).replace('##', '')
    split_title = title.split(' ')
    attempt_title = ''
    attempt_index = 0
    for word in split_title:
        attempt_title += word
        if limit_title.startswith(attempt_title.lower()):
            attempt_index += 1
    new_title = ' '.join(split_title[:attempt_index + 1])
    news_dict[id_] = new_title

f_main = open(main_data, 'w')
f_sub = open(sub_data, 'w')
main_pos_neg = [0, 0]
sub_pos_neg = [0, 0]

for interaction in interaction_list:
    interaction_features = interaction.split('\t')
    histories = interaction_features[3].split(' ')
    if interaction_features[4].endswith('\n'):
        interaction_features[4] = interaction_features[4][:-1]
    impressions = interaction_features[4].split(' ')

    histories = histories[::-1][:MAX_USER_HIS][::-1]
    history_news = []
    for news_id in histories:
        if news_id in news_dict:
            history_news.append(news_dict[news_id])
    if not history_news:
        print(interaction)
        continue
    user = ' '.join(history_news)
    # user = ' '.join([news_dict[news_id] for news_id in histories[::-1][:15][::-1]])

    sub_count = int(len(impressions) * sub_ratio)
    main_count = len(impressions) - sub_count
    for index, impression in enumerate(impressions):
        news_id, label = impression.split('-')
        if news_id not in news_dict:
            continue
        line = '\t'.join([user, news_dict[news_id], label]) + '\n'
        if index < main_count:
            f_main.write(line)
            main_pos_neg[int(label)] += 1
        else:
            f_sub.write(line)
            sub_pos_neg[int(label)] += 1

f_main.close()
f_sub.close()

print(main_pos_neg)
print(sub_pos_neg)
