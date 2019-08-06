import re
import jieba
import json


# Bulid function with regular expression to get words and numbers
def token(string):
    return ''.join(re.findall('[\w|\d]+', string))


def get_two_gram_count(path):
    two_gram_count = {}

    with open(path) as file:
        for line in file:
            # To skip the article title line
            if line.strip().startswith('</doc>') or line.strip().startswith('<doc'): continue
            if line:
                # To keep just words and numbers
                line = token(line)
                all_tokens = list(jieba.cut(line.strip()))

                # Get 2_gram words list
                all_2_gram_words = [''.join(all_tokens[i:i + 2]) for i in range(len(all_tokens[:-2]))]

                # Get 2_gram words frequencies
                for w in all_2_gram_words:
                    if w in two_gram_count:
                        two_gram_count[w] += 1
                    else:
                        two_gram_count[w] = 1

    return two_gram_count


def main():
    path = './data/wiki_corpus_simple.txt'
    two_gram_count = get_two_gram_count(path)

    # Write file
    with open('./data/assignment_2_gram.json', 'w', encoding='utf-8') as f:
        json.dump(two_gram_count, f)


if __name__ == '__main__':
    main()
