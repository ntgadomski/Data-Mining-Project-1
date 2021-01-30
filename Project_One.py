#Authors: Nickolas Gadomski, Colby Nicoletti, Carlos Martinez
#Purpose: Research and manipulate a CSV file data set and graph the results

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
#from PIL import Image
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#nltk.download('stopwords')
#np.set_printoptions(threshold=np.inf)

##########Initial Manipulation and Calculations of Data########################

#Reads and Chunks raw_data(CSV) into 10000 chuncksize
for raw_data in pd.read_csv('rt_reviews.csv', chunksize=10000, usecols=range(0, 2), header = 0):
    print()

#Global Variables
X = raw_data.Review
# Y = raw_data.Freshness
#raw_data = pd.read_csv('small_rt.csv')

#Sets up data frame and creates Freshness and Review columns for dataset
df = pd.DataFrame(raw_data, columns = ['Freshness', 'Review'])
print(df)

length_df = len(df)

#counts amount of chars within the review
countVectorizer = CountVectorizer(analyzer= 'char')
tf = countVectorizer.fit_transform(X)
tf_df = pd.DataFrame(tf.toarray(),
                      columns= countVectorizer.get_feature_names())
print(tf_df)

#Reads all characters in the review column and calculates average char amount
res = sum(map(len, df.iloc[:,1]))/float(len(df.iloc[:,1]))
print('Average character length in each review:\n', res)

#Initializes Freshness column with matching string 'fresh'
fresh = len(df[df['Freshness'].str.match('fresh')])
print('Total number of fresh reviews:\n', fresh)

#Initializes Freshness column with matching string 'rotten'
rotten = len(df[df['Freshness'].str.match('rotten')])
print('Total number of rotten reviews:\n', rotten)

#Calculates fresh and rotten percentages from the Freshness column 
percent_fresh = (fresh / length_df) * 100 
percent_rotten = (rotten / length_df) * 100
print(percent_fresh, '% of reviews are fresh.')
print(percent_rotten, '% of reviews are rotten.')

##########Plotting Initial Data Calculations###################################

#Creates Groups that will be graphed on Bar Chart
n_groups = 1

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

#Creates a rectangle Bar Chart that utilize percent_fresh variable
rects1 = plt.bar(index, percent_fresh, bar_width, color='g', 
    alpha= 0.8)

#Creates a rectangle Bar Chart that utilize percent_rotten variable
rects2 = plt.bar(index + bar_width, percent_rotten, bar_width, color='y', 
    alpha= 0.8)

#Creates X and Y labels for graph
plt.xlabel('Fresh                                                   Rotten\nFreshness')
plt.ylabel('Percentage')
plt.title('Fresh Vs. Rotten')
plt.xticks([])
plt.legend()

#Controls Bar Chart heights and displys Percentage above bar
for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 0.99*height,
            '%d' % int(height) + "%", ha='center', va='bottom')
for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 0.99*height,
            '%d' % int(height) + "%", ha='center', va='bottom')

plt.tight_layout()
plt.show()

#################Splitting Data into Naive Bayes###############################

#Splitting of the Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(df['Review'], 
                                                    df['Freshness'], 
                                                    test_size=0.2, random_state = 42)
#Initializing Frequency Vectorizer 
#Params: max_features=15000, stop_words='english', lowercase=True 
vectorizer = TfidfVectorizer(max_features=15000, stop_words='english',
                             lowercase=True)
                            

#Fits the X_train data(Reviews) to an n-gram array
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
print('tfidf train shape:', X_train_tfidf.shape)
print('tfidf train type:', X_train_tfidf.dtype)

# use the same as above to transform X_test
X_test_tfidf = vectorizer.transform(X_test).toarray()
print('tfidf test:', X_test_tfidf.shape)

#Initializes Naive Bayes algorithm
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

#test results
predicted = clf.predict(X_test_tfidf)

#Shows the accuracy of the n-gram comparison through Naive bayes
acc = metrics.accuracy_score(y_test, predicted)
print('accuracy is: ', acc*100)



############Plotting Frequency Uni-Gram Data###########################################

#Fit transform X_test dataset
X = vectorizer.fit_transform(X_test)

# zipping actual words and sum of their Tfidf
features_rank = list(zip(vectorizer.get_feature_names(), [x[0] for x in X.sum(axis=0).T.tolist()]))

# sorting
features_rank = np.array(sorted(features_rank, key=lambda x:x[1], reverse=True))
# print(features_rank[5])
# print(features_rank[6])

#Plot Frequency of Uni-Grams
n = 5
plt.figure(figsize=(5, 10))
plt.barh(-np.arange(n), features_rank[:n, 1].astype(float), height=.8)
plt.title('Most Frequent Uni-Grams from CSV')
plt.xlabel("Frequency of Uni-Grams")
plt.ylabel("5 Most Frequent Uni-Grams")
plt.yticks(ticks=-np.arange(n), labels=features_rank[:n, 0])

#modify to make different graphs
#stop_words = frozenset(["full review in", "word2","word3"])

##################Plotting Count Bi-Gram Data##################################

#Create Count Vectorizer for graphing N-Gram Datasets
#Params: stop_words = 'english', lowercase=True, max_features=150400, ngram_range=(2,2)
countVectorizer = CountVectorizer(stop_words = 'english',
                             lowercase=True, max_features=15000, ngram_range=(2,2))

#Create Variables that will read Freshness column to find rotten or fresh
rottenReviews = df.loc[df['Freshness']=='rotten']
freshReviews = df.loc[df['Freshness']=='fresh']

#Split Dataset to be graphed
X_train, X_test, y_train, y_test = train_test_split(rottenReviews['Review'], 
                                                    rottenReviews['Freshness'], 
                                                    test_size=0.2, random_state = 42)
#Fit transform X_test dataset
X = countVectorizer.fit_transform(X_test)

# zipping actual words and sum of their Count
features_rank = list(zip(countVectorizer.get_feature_names(), [x[0] for x in X.sum(axis=0).T.tolist()]))

# sorting
features_rank = np.array(sorted(features_rank, key=lambda x:x[1], reverse=True))
# print(features_rank[5])
# print(features_rank[6])

#Plot Rotten word Bi - grams
n = 5
plt.figure(figsize=(5, 10))
plt.barh(-np.arange(n), features_rank[:n, 1].astype(float), height=.8)
plt.title("Rotten words")
plt.xlabel("Frequency of Bi-Grams")
plt.ylabel("5 Most Frequent Bi-Grams")
plt.yticks(ticks=-np.arange(n), labels=features_rank[:n, 0])

#Split Dataset to be graphed
X_train, X_test, y_train, y_test = train_test_split(freshReviews['Review'], 
                                                    freshReviews['Freshness'], 
                                                      test_size=0.2, random_state = 42)
#Fit transform X_test dataset
X = countVectorizer.fit_transform(X_test)

# zipping actual words and sum of their Count
features_rank = list(zip(countVectorizer.get_feature_names(), [x[0] for x in X.sum(axis=0).T.tolist()]))

# sorting
features_rank = np.array(sorted(features_rank, key=lambda x:x[1], reverse=True))
# print(features_rank[5])
# print(features_rank[6])

#Plot Fresh word Bi - Grams
n = 5
plt.figure(figsize=(5, 10))
plt.barh(-np.arange(n), features_rank[:n, 1].astype(float), height=.8)
plt.title("Fresh words")
plt.xlabel("Frequency of Bi-Grams")
plt.ylabel("5 Most Frequent Bi-Grams")
plt.yticks(ticks=-np.arange(n), labels=features_rank[:n, 0])

####################Plotting Count Tri-Gram Data###############################

#Create Count Vectorizer for graphing N-Gram Datasets
#Params: stop_words = 'english', lowercase=True, max_features=15000, ngram_range=(2,2)
countVectorizer = CountVectorizer(stop_words = 'english',
                             lowercase=True, max_features=15000, ngram_range=(3,3))

#Create Variables that will read Freshness column to find rotten or fresh
rottenReviews = df.loc[df['Freshness']=='rotten']
freshReviews = df.loc[df['Freshness']=='fresh']

#Split Dataset to be graphed
X_train, X_test, y_train, y_test = train_test_split(rottenReviews['Review'], 
                                                    rottenReviews['Freshness'], 
                                                    test_size=0.2, random_state = 42)
#Fit transform X_test dataset
X = countVectorizer.fit_transform(X_test)

# zipping actual words and sum of their Count
features_rank = list(zip(countVectorizer.get_feature_names(), [x[0] for x in X.sum(axis=0).T.tolist()]))

# sorting
features_rank = np.array(sorted(features_rank, key=lambda x:x[1], reverse=True))
# print(features_rank[5])
# print(features_rank[6])

#Plot Rotten word Tri - Grams
n = 5
plt.figure(figsize=(5, 10))
plt.barh(-np.arange(n), features_rank[:n, 1].astype(float), height=.8)
plt.title("Rotten words")
plt.xlabel("Frequency of Tri-Grams")
plt.ylabel("5 Most Frequent Tri-Grams")
plt.yticks(ticks=-np.arange(n), labels=features_rank[:n, 0])

#Split Dataset to be graphed
X_train, X_test, y_train, y_test = train_test_split(freshReviews['Review'], 
                                                    freshReviews['Freshness'], 
                                                      test_size=0.2, random_state = 42)
#Fit transform X_test dataset
X = countVectorizer.fit_transform(X_test)

# zipping actual words and sum of their Count
features_rank = list(zip(countVectorizer.get_feature_names(), [x[0] for x in X.sum(axis=0).T.tolist()]))

# sorting
features_rank = np.array(sorted(features_rank, key=lambda x:x[1], reverse=True))
# print(features_rank[5])
# print(features_rank[6])

#Plot Fresh word Tri - Grams
n = 5
plt.figure(figsize=(5, 10))
plt.barh(-np.arange(n), features_rank[:n, 1].astype(float), height=.8)
plt.title("Fresh words")
plt.xlabel("Frequency of Tri-Grams")
plt.ylabel("5 Most Frequent Tri-Grams")
plt.yticks(ticks=-np.arange(n), labels=features_rank[:n, 0])


