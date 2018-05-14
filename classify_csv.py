def predict(list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.externals import joblib
   
    vect = joblib.load('tfidf.pkl')
    dict = joblib.load('dict.pkl')
    #feature_list = vect.get_feature_names()
    loaded_model = joblib.load('linear_svc_doc_clf_model.pkl')
    #indices = np.argsort(loaded_model.coef_['category_id'])
    #list =['hgjjhgj hghjgjghjgh']
    text = vect.transform(list)
    
    predicted_IDs = loaded_model.predict(text)
    for ID in predicted_IDs:
        # print(dict[ID])
        return dict[ID]

def main():
    predict(['Test'])
    
if __name__== "__main__":
    main()
