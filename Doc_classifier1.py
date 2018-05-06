
# coding: utf-8

# In[7]:


import pandas as pd
column_names = ['Label','Content']
temp = pd.read_csv('orig.csv', names = column_names,header=None,chunksize=1000)
df = pd.concat(temp, ignore_index=True)
#df_second = df.iloc[:,1:2]
df.head(10)


# In[8]:


'''for i in range(len(df['Label'])):
  if(df['Label'][i].isupper()):
    df['category_id'] = df['Label'].factorize()[0]
'''    
df['category_id'] = df['Label'].factorize()[0]


# In[9]:


category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)

#category_id_df.head(66)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', )
features = tfidf.fit_transform(df.Content.astype('str'))
labels = df.category_id
features.shape


# In[15]:


import matplotlib.pyplot as plt
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.12, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
import seaborn as sns
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[ ]:


'''from sklearn.feature_selection import chi2
import numpy as np
N = 5
for Label, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  #print("# '{}':".format(Label))
  #print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
'''  


# In[5]:


from sklearn.model_selection import train_test_split
'''from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
'''
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(df['Content'], df['Label'], random_state = 0)
#count_vect = CountVectorizer()
logistic_reg_model = LogisticRegression()
CV=5
cv_df = pd.DataFrame(index=range(CV))
entries = []
models = [
#   RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    LogisticRegression(random_state=0)]

for model in models:
   model_name = model.__class__.__name__
   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=5)
   for fold_idx, accuracy in enumerate(accuracies):
      entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
cv_df.groupby('model_name').accuracy.mean()

#logistic_reg_model.fit(X_train, y_train)

#X_train_counts = count_vect.fit_transform(X_train.astype(str))
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=5)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
cv_df.groupby('model_name').accuracy.mean()


# In[ ]:


start = 240
correct = 0
print(start)
while(start < 260):
    print(clf.predict(count_vect.transform(df['Content'][start] )) )
    start+=1
#    if(df['Label'][start] == clf.predict(count_vect.transform(df['Content'][start] ))):
#        correct +=1
#        start+=1
            
#print(correct)

