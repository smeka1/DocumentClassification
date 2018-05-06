def predict():
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.externals import joblib
    
    column_names = ['Content']
    temp = pd.read_csv('test100.csv', names = column_names,header=None,chunksize=1000)
    df = pd.concat(temp, ignore_index=True)
    
    vect = joblib.load('tfidf.pkl')
    dict = joblib.load('dict.pkl')
    #feature_list = vect.get_feature_names()
    loaded_model = joblib.load('linear_svc_doc_clf_model.pkl')
    #indices = np.argsort(loaded_model.coef_['category_id'])
    text = vect.transform(df['Content'])
    i =1
    predicted_IDs = loaded_model.predict(text)
    for ID in predicted_IDs:
        print(dict[ID])
    
def main():
    predict()
    
if __name__== "__main__":
    main()
