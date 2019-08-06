# ML_AI_NLP_DL
>The farther backward you can look, the farther forward you are likely to see.<br>
_(Winston Churchill)_
----
## Solution if GitHub load .ipynb file failed
[Open the .ipynb file on this website](https://nbviewer.jupyter.org/)
## Table of Contents
### Week 01
- Course Background
    - Three problems to solve
        1. Confidence
        2. Capacity
        3. Continuation
- AI Introduction
    - Why Natural Language is difficult
        1. Text is Logic
        2. Diversity & Heterogeneity
        3. Unstructured
    - AI Paradigm
        1. Search Policy Bases(BFS, DFS, Map App, Decision System)
        2. Rule Based: from **Rules** to **Data Driving**
        3. Mathematical or Analytic Based
        4. Probability Based
        5. Machine Learning(deep learning) Based
- Search Policy, Syntax Tree
- [Recode Lecture 01](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_01_PatternMatch/LectureCode_01.ipynb)
    1. [x] Map Search
    2. [x] Sentence Generation    
- [Assignment 01: Pattern Match](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_01_PatternMatch/Assignment_01.ipynb)
    1. [x] Chat Bot Using Pattern: Sample Eliza
- [Code of BFS and DFS](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_01_PatternMatch/Search.py)
----
### Week 02
- Language Model
    - Basic Crawler
    - Regular Expression
    - Build Unigram(1_gram) model
    - Build Bigram(2_gram) model
- [Recode Lecture 02](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_02_LanguageModel/LectureCode_02.ipynb)
    1. [x] Language Model
- [Assignment 02: Language Model](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_02_LanguageModel/Assignment_02.ipynb)
    1. [x] Prepare Wikipedia Corpus
    2. [x] Build Unigram(1_gram) model
    3. [x] Build Bigram(2_gram) model
    4. [x] Turing-Good Estimator smooth method
----
### Week 03
- Review
    - Syntax Tree: Rule Based System
    - BFS Search, DFS Search: Graph Based System
    - Language Model: Probability Based System
- Machine Learning Based System
    - Data Driven
    - Linear Regression
    - Classification
    - Plus: Logic Reasoning System
- The Problem of Search
    - Heuristic Function
        1. The start state
        2. The goal state
        3. The successors
        4. The strategy that determines the order in which we search
    - Search Problem, Search Tree    
- How to make search faster
    - Heuristic Search: A* Search
    - [ ] Dynamic Programming
- [Recode Lecture 03](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_03_SearchAgent/LectureCode_03.ipynb)
    - [x] Best First Search: Basic Navigator for Chinese Cities
    - [X] Gradient Descent: Titanic Machine Learning
- [Assignment 03: Search Agent](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_03_SearchAgent/Assignment_03.ipynb)
    - [x] Crawler Beijing Subway Data
    - [x] Subway Route Visualization
    - [x] Build Search Agent
    - [x] Implement Various Search Strategy
    - [ ] Strategy: by_way
----
### Week 04
- Why NLP
    - Information Chaos
    - Unstructured
    - Discrete
    - Unconventional
    - OOV
- Utilities for NLP
    - Similarity: Edit Distance, Word Distance
    - Key words
    - Name Entity Recognition
    - Dependency Parsing
    - Topic Model
- Dynamic Programming
    - Rod Cutting Problem
    - Edit Distance Problem
    - [ ] Key Characteristics for Dynamic Programming
    - [ ] The Travel Salesman Problem
- [Recode Lecture 04](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_04_DynamicProgramming/LectureCode_04.ipynb)
    - [x] Rod Cutting Problems
    - [x] Python Decorator
    - [x] Edit Distance (Levenshtein)
- [Assignment 04: Dynamic Programming](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_04_DynamicProgramming/Assignment_04.ipynb)
    - [x] Solution Parse of Edit Distance
    - [ ] K-Person-Salesman Problem
#### AI English Enhancement Lesson 01
- [Module_01](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/AI_English_Enhancement_01/Module_1-Academic_English_Reading.pdf)
- Content
    1. Language Proficiency Test
    2. Choose the Appropriate Learning Material
    3. Purpose of Academic Reading
    4. Reading Strategies for Academic Texts
    5. Reading Fluency Technique: Chunking Text
----
### Week 06
- Keywords
    - Get related words by word2vec
- TF_IDF
    - Term Frequency
    - Inverse Document Frequency
- TF_IDF Vectorized
    - Scikit-Learning TF_IDF
    - Simplest Classification Model
- WordCloud
- Boolean Search
    - To process large document collections quickly
    - To allow more flexible matching operations
    - To allow ranked retrieval
- PageRank
    - Ranking using TF_IDF
- [Recode Lecture 06](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_06_TFIDF/LectureCode_06.ipynb)
    - [x] Similar Words
        1. Regular Expression
        2. Jieba Cut Words
        3. Gensim Word2Vec
    - [x] TF_IDF Key Words
        1. Term Frequency
        2. Inverse Document Frequency
        3. Calculate TF_IDF
    - [x] WordCloud
        1. Plot WordCloud
        2. Plot with image mask
    - [x] TF_IDF Vectorized
        1. Cosine Similarity
        2. TF_IDF Vectorized with Sklearn
        3. Search the similar document
    - [x] Boolean Search
        1. Search Engine with TF_IDF
        2. PageRank
----
### Week 07
- Machine Learning
    - Background
    - Main Methodologies
    - Current Trends
- Under-fitting and Over-fitting
    - Bias and Variance
    - Model Capacity
    - Under-fitting and Over-fitting
- Train set, test set, validation set
    - Influence of dataset
    - The relation of train, test and validation
- [Assignment 07](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_07_MachineLearning/Assignment_07.md)
    - [x] Difference between regression and classification
    - [x] Model evaluation index
    - [x] Reason for over-fitting
    - [x] Reason for under-fitting
----
### Week 08
- Lazy Learning and Eager Learning
    - Target function will be approximately locally
    - Dataset with few attributes
- Outlier
    - What`s outlier and how to detect
    - Percentile
- Bias and Variance
- Train, Validation and Test
- Supervised Learning
    - Classification and Regression
    - Logistic Regression
    - Bayes Model
    - KNN Model
    - SVM Model
    - Decision Tree
- Scikit-Learning
- Unsupervised Learning
    - K-means
    - Hierarchy clustering
    - Embedding clustering
- Semi-Supervised Learning
    - Activate Learning
    - Machine Learning + Search Policy
- Importance of Pre_processing
    - Normalization and Scalar
- [Assignment_08](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_08_LR/Assignment_08.ipynb)
    - [x] Basic Concept
    - [x] 新华社抄袭文本判断
    - [x] [Logistic_Regression](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_08_LR/Logistic_Regression.ipynb)
----
### Week 09
- Cross Entropy
- Multi-Classification
- Softmax
- K-Nearest Neighbors
    - If we want write a program solve it, it`s simple.
    - 1-nearest neighbor classifier / regression.
    - Weighted nearest neighbor classifier / regression.
    - Maybe all the machine learning problem could use this method.
        - Dimension reduction
        - Outlier
        - Memory Consuming
        - Timing Consuming in high dimension
- Support Vector Machine
    - Kernel Function
        - Polynomial(homogeneous)
        - Polynomial(inhomogeneous)
        - Gaussian radial basis function
        - Hyperbolic tangent
    - Unbalance Problem
- Importance of Pre-processing
    - Balance Analysis
    - Remove Noise
    - Remove Col-linearity
    - Rescale Inputs
- [Assignment_09](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_09_ClassicalModels_1/news_plagiarism_detection.py)
    - [x] News plagiarism detection
----
### Week 10
- Bayesian Classifier
- Decision Tree
- Random Forest
    - Create a 'bootstraped' data set
    - Predicate using the forest
    - Before predicating we use Out-Of-Bagging sample to evaluate our model
    - Using the evaluated result, we could choose the right variables number to be chosen in Random Forest.
- XGBoost
- K-means Cluster
- Embedding Cluster
- Semi Supervised & Activate Learning
- [Record Lecture 10](https://github.com/ProGe666/ML_AI_NLP_DL/blob/master/Week_10_ClassicalModels_2/LectureCode_10.ipynb)
