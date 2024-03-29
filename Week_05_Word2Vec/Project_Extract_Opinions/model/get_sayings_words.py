from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import re
import jieba
import os
from pyltp import Segmentor
from pyltp import Postagger
from gensim.models import KeyedVectors


# Function to get token from string
def token(string):
    return re.findall(r'[\d|\w]+', string)


# Function to cut words with jieba
def cut_words_jieba(string):
    return ' '.join(jieba.cut(string))


# Function to cut words with ltp
def cut_words_ltp(sentence):
    # Set pyltp cut words model path
    LTP_DATA_DIR = '../ltp_data_v3.4.0'
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

    # Init segmentor
    segmentor = Segmentor()

    # Load pyltp cut words model
    segmentor.load(cws_model_path)

    # Cut words
    words = segmentor.segment(sentence)

    # Close model
    segmentor.release()

    # Return cut words string
    return ' '.join([word for word in words])


# Function to process corpus
def corpus_processing(corpus_path):
    # Load corpus
    df = pd.read_csv(corpus_path)

    # Get content
    news = df['content'].tolist()

    # Filter with regular expression
    news = [token(str(n)) for n in news]
    news = [''.join(n) for n in news]

    # # Cut words with jieba
    # news = [cut_words_jieba(n) for n in news]
    # with open('./model_data/total_news_sentences_cut.txt', 'w') as f:
    #     for n in news:
    #         f.write(n + '\n')

    # Cut words with ltp
    news = [cut_words_ltp(n) for n in news]

    # Save result to txt file
    with open('./model_data/total_news_sentences_cut_ltp.txt', 'w') as f:
        for n in news:
            f.write(n + '\n')


# Function to get word_2_vector from cut sentence file
def get_word2vec(cut_sentence_path):
    news_word2vec = Word2Vec(LineSentence(cut_sentence_path), sg=0, min_count=10, size=100, window=5, workers=8)
    # Save model
    news_word2vec.wv.save_word2vec_format('./model_data/news_word2vec_model_ltp.txt', binary=False)
    return news_word2vec


def get_related_words(init_words, model, max_size, top_n):
    """
    @ init_words: initial words we already know
    @ model: the word2vec model
    @ max_size: the maximum number of words need to see
    @ top_n: the number of top similar words
    """

    # Init unseen words list
    unseen_list = init_words

    # Init seen words dict
    seen = defaultdict(int)

    # Init sub nodes dict
    sub_nodes_dic = defaultdict(list)

    # Scan unseen words list if length of seen words dict less than max_size
    while unseen_list and len(seen) < max_size:

        # Get first word in unseen words list
        node = unseen_list.pop(0)

        # Get sub nodes directly if in dict
        if node in sub_nodes_dic:
            sub_nodes = sub_nodes_dic[node]

        else:
            # Get top_n similar words for first word by word2vec model
            sub_nodes = [w for w, s in model.most_similar(node, topn=top_n)]

            # Save result to sub nodes dict
            sub_nodes_dic[node] = sub_nodes

        # Add similar words result to unseen words list
        unseen_list += sub_nodes

        # Save current seen word and increase 1 weight
        seen[node] += 1  # could be weighted by others

    # Optimize with Similarity Weight
    for word, value in seen.items():
        weight = model.similarity(init_words[0], word)

        seen[word] = value * weight

    # Sort seen words dict by words weight
    seen_rank = sorted(seen.items(), key=lambda x: x[1], reverse=True)

    # Return sorted list
    return [w for w, s in seen_rank]


# Function to optimize with stop words
def stop_words_opt(stop_words_path, related_words):
    # Load stop words list
    stop_list = []
    with open(stop_words_path, 'r') as f:
        for line in f:
            stop_list.append(line[0])

    # Init result list
    words_list = []

    # Filter with stop words
    for w in related_words:
        if w not in stop_list:
            words_list.append(w)

    return words_list


# Function to optimize with pyltp postags
def postags_opt(words):
    # Set pyltp postagger model path
    LTP_DATA_DIR = '../ltp_data_v3.4.0'
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')

    # Init postagger
    postagger = Postagger()

    # Load model
    postagger.load(pos_model_path)

    # Get postags
    postags = postagger.postag(words)

    # Close postagger
    postagger.release()

    postags = list(postags)

    # Init result list
    saying_words = []

    # Filter with tag 'verb'
    for index, tag in enumerate(postags):
        if tag == 'v':
            saying_words.append(words[index])

    return saying_words


def main():
    corpus_path = '../corpus/data/total_news_corpus.csv'
    corpus_processing(corpus_path)

    cut_sentence_path = './model_data/total_news_sentences_cut_ltp.txt'
    news_word2vec = get_word2vec(cut_sentence_path)

    # Load word2vec model
    news_word2vec = KeyedVectors.load_word2vec_format('./news_word2vec_model_ltp.txt', binary=False)

    # Get related words
    related_words = get_related_words(['说', '表示'], news_word2vec, max_size=10000, top_n=50)

    # Optimize with stop words list
    stop_words_path = './model_data/chinese_stop_words.txt'
    words_after_stop_words = stop_words_opt(stop_words_path, related_words)

    # Optimize with pyltp postags
    related_top_400 = words_after_stop_words[: 400]
    saying_words = postags_opt(related_top_400)

    print(saying_words)
    print(len(saying_words))

    # Save saying words
    with open('/model_data/saying_words.txt', 'w') as f:
        for n in saying_words:
            f.write(n + '\n')


if __name__ == '__main__':
    main()
