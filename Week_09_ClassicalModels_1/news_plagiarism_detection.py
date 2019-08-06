import pandas as pd
import re
import jieba
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle


# Function to load source file
def load_data(news_path):
    # Load file
    df_source = pd.read_csv(news_path, encoding='gb18030')

    # Drop news which content or source is empty
    df_source = df_source.dropna(subset=['source', 'content'])

    # fill other empty values with ''
    df_source = df_source.fillna('')

    return df_source


# function to cut words with jieba
def cut_content(string):
    clean_str = ''.join(re.findall(r'[\d|\w]+', string)).strip().replace(u'\n', u'')
    return ' '.join(jieba.cut(clean_str))


# Function to preprocess source file
def pre_process(df_source):
    # Add a new column to save content after cutting words
    df_source['content_cut'] = df_source['content'].apply(cut_content)

    # Add label for each news, 1 for 新华社， 0 for others
    df_source['label'] = df_source['source'].apply(lambda x: 1 if any(pd.Series(x).str.contains('新华')) else 0)

    # Save result to new CSV file
    df_news = df_source[['id', 'source', 'content', 'content_cut', 'label']]
    # df_news.to_csv('./data/news_cut.csv', encoding='utf-8', index=False)

    return df_news


# Function to take samples
def take_sample(df_news):
    # Get positive samples = 8000
    df_pos = df_news[df_news['label'] == 1].sample(8000).reset_index(drop=True)

    # Get negative samples = 8000
    df_neg = df_news[df_news['label'] == 0].sample(8000).reset_index(drop=True)

    # Combine pos and neg samples
    df_sample = pd.concat([df_pos, df_neg], axis=0)

    # Shuffle the samples
    df_sample = shuffle(df_sample).reset_index(drop=True)

    return df_sample


# Vectorize news with TFIDF
def tfidf_vect(df_sample):
    X_words = df_sample['content_cut']

    # Vectorize news with TFIDF
    # Init vectorizer, set max_feature=10000
    vectorizer = TfidfVectorizer(token_pattern=u'(?u)\w+', max_features=10000)

    # Fit
    X = vectorizer.fit_transform(X_words.values.astype('U'))

    return X


# Function to get performance of model
def get_performance(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print('percision is: {}'.format(precision_score(y_test, y_pred)))
    print('recall is: {}'.format(recall_score(y_test, y_pred)))
    print('roc_auc is: {}'.format(roc_auc_score(y_test, y_pred)))


# Build function to create model with GridSearchCV
def build_model(X_train, X_test, y_train, y_test, model_dict):
    gscv = {}

    for model_name, (model, param_range) in model_dict.items():
        print('Start Training {} ...'.format(model_name))

        # Time
        start = time.time()

        # 5 Folder Cross Validation and Grid Search
        clf = GridSearchCV(estimator=model,
                           param_grid=param_range,
                           cv=5,
                           scoring='roc_auc',
                           refit=True)

        clf.fit(X_train, y_train)

        end = time.time()
        duration = end - start

        get_performance(clf, X_test, y_test)

        print('The best paramter：{}'.format(clf.best_params_))
        print('Total Training Time: {:.4f}s'.format(duration))
        print('###########################################')

        gscv[model_name] = clf

    return gscv


def main():
    news_path = './data/sqlResult.csv'
    df_source = load_data(news_path)
    df_news = pre_process(df_source)
    df_sample = take_sample(df_news)

    # Set X and y
    X = tfidf_vect(df_sample)
    y = df_sample['label']

    # Split data into train data and test data with ratio 4:1
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=1)

    # Set model dict
    model_dict = {
        'MNB': (MultinomialNB(), {}),
        'LR': (LogisticRegression(solver='lbfgs'), {}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
        'DT': (DecisionTreeClassifier(), {'max_depth': [3, 5, 7]}),
        'SVM': (SVC(kernel='sigmoid', gamma='auto'), {})
    }

    gscv = build_model(X_train, X_test, y_train, y_test, model_dict)

    # # Save Model
    # for model_name, model in gscv.items():
    #     with open(model_name + '.pkl', 'wb') as f:
    #         pickle.dump(model, f)
    #
    # # Load Model
    # with open('LR.pkl', 'rb') as f:
    #     lr_clf = pickle.load(f)


if __name__ == '__main__':
    main()
