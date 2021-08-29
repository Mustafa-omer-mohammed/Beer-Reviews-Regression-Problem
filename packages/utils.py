import os
import sys
module_path = os.path.abspath(os.path.join('project/packages/utils.py'))
# if not os.path.exists(module_path):
#     os.makedirs(module_path)

import pandas as pd # algebraic computations , read and write to csv 

import numpy as np # linear algebra and arrays muniplication

############  importing the data visualization libraries :##############

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm # Creating a normaly distributed curve

# Importing a model to split the training set from the evaluation set
from sklearn. model_selection import train_test_split 

##################### Importing the Regression Models####################
from sklearn. ensemble import RandomForestRegressor # Random forest regressor model
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

############# R2 as the unit of evaluation measure #################
from sklearn. metrics import r2_score                

############  RandomizedSearchCV #################
from sklearn.model_selection import RandomizedSearchCV


################## Tuxtual data prprocessing ##################
from sklearn. feature_extraction.text import TfidfVectorizer 
import warnings
warnings.filterwarnings("ignore")
from os import path
from PIL import Image
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as sw
import re
import string


################ Defining Functions ##############

###### Train and test split 


def train_val_test(Dropped_columns , DataFrame ):
    df_drop = DataFrame.drop(columns=Dropped_columns)
    # define the mask for the training/validation samples (those with a Quality , the others will belong to the evaluation set )
    train_valid_mask = ~df_drop["overall"].isna()
    # extract the feature names (for later use)
    feature_names = df_drop[train_valid_mask].drop(columns=["overall"]).columns
    X = df_drop.drop(columns=["overall"]).values
    y = df_drop["overall"].values

    X_train_valid = X[train_valid_mask]
    y_train_valid = y[train_valid_mask]
    X_test = X[~train_valid_mask]
    y_test = y[~train_valid_mask]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,y_train_valid, shuffle=True, random_state=42)
    index = DataFrame[~train_valid_mask].index
    return X_train, X_valid, y_train, y_valid ,X_train_valid ,y_train_valid,X_test,y_test,feature_names ,index


##### Training models Functions 

def make_regression(reg , X_train , y_train ,X_valid , y_valid) :
    reg.fit(X_train , y_train)
    score = r2_score(y_valid, reg.predict(X_valid) )
    print ( f'the R2 Score for the {str(reg)} is { score:.4f}' )


##### Randomize Grid Search CV :


def RandomizeGridSearch (reg , param ,X_train_valid , y_train_valid , n_iter,cv):
    model_RG =  RandomizedSearchCV(estimator = reg, param_distributions = param, scoring= 'r2' , n_iter = n_iter, cv = cv, verbose=2, random_state=42, n_jobs = -1)
    model_RG.fit(X_train_valid, y_train_valid)
    print(model_RG.best_score_)
    print(model_RG.best_estimator_)
    print(model_RG.best_params_ )
    
    return model_RG.best_estimator_


# #### Fitting the model overall Training Data and Saving the best model Predicitons (on X_test ) : 


def print_results (besr_reg ,X_train_valid, y_train_valid,X_test, outputname , index ):
    besr_reg.fit(X_train_valid, y_train_valid)
    y_pred = besr_reg.predict(X_test)
    output =  str(outputname) + ".csv"
    pd.DataFrame(y_pred, index=index).to_csv(output , index_label="Id", header=["Predicted"])
    print(f" a csv file was made with the name {output}")


##### Visualization Function 


# defining a funcion to visualize the distribution of the attributes 
# hence the variance in the distribution of the values may vary from one attribute from the others .

def data_visualization_histogram(attribute,dataframe,n):
    # where "attribute" is the attribute we want to visualize it's distribution , "n" number of the most frequent to be displaied 

    attribute_perc = dataframe[attribute].value_counts().head(n)
    print(dataframe[attribute].value_counts().head(n))
    # the ratio with respect to the column length 

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,10), sharey=True)
    # for better visualization we should represent the text on the vertical axis and use horisantale

    ax.barh(y=attribute_perc.index, width=attribute_perc.values, color='blue', edgecolor='white') 
    ax.set_title(f'Occurance of {attribute} in the Dataset', fontsize=20)
    ax.set_xlabel('Occurance in the Dataset')
    ax.set_ylabel(attribute)


##### Creat Word Cloud 

def show_wordcloud (data,  stopwords , title = None ):
    wordcloud = WordCloud(background_color='white', stopwords = stopwords,max_words=100, max_font_size=40, scale=3, random_state=1).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    wordcloud.to_file(f"./{title}.png")
    plt.show()


##### Text Cleaing Function 


def cleaining_Text(lemmatizer, document):
    document = str(document)
    lemmas = []
    re_digit = re.compile("[0-9]") # regular expression to filter digit tokens
    for t in word_tokenize(document):
        t = t.strip()
        lemma = lemmatizer.lemmatize(t)
    # remove tokens with only punctuation chars and digits
        if lemma not in string.punctuation and len(lemma) > 3 and len(lemma) < 16 and not re_digit.match(lemma):
            lemmas.append(lemma)
    lemmatized_output = ' '.join([w for w in lemmas])
    return lemmatized_output


#####  The Nth most frequent TF-IDF Terms
global vectorizer

def selecting_terms(N , tfidf_vectorizer_vectors , vectorizer) :
    # To get the value of each term along the whole collection we must sum the values of tf_idf along the column [axis=0]
    # tfidf_vectorizer_vectors.sum(axis=0)        # the result is 1 row due to the aggregation and  62641 columns 1 for each term
                                   
    # since this gives us a matrix we can converte it to a list and select the fisrt element [0] 
    tf_list = tfidf_vectorizer_vectors.sum(axis=0).tolist()[0] 
    
    fea_names = vectorizer.get_feature_names()   # GET the features names extracted by the vectorizer.

    feat_value = zip(fea_names, tf_list)         # to map the feature name to the values we use zip method 

    freq = sorted(feat_value,key=lambda x: x[1] , reverse=True)[:N] # since we need the decsinding orded reverse=true 
    
    return freq


##### create TF-IDF Data frame  



# To select the MOST FREQUENT we can make a mask from tfidf_vectorizer_vectors which has binary values about words are present or not
def word_dataframe(freq , tfidf_vectorizer_vectors , vectorizer , dataframe):
    # mask to be use d to filter columns in wpm (only keeps the ones for the N most frequent words)
    words = [ word for word, _ in freq ] # the second part of the tuple is the value , we need to select only the first part word,_

    mask = [ w in words for w in vectorizer. get_feature_names() ] # the mask will filter only the most frequent words

    words_ = [ w for w in vectorizer. get_feature_names() if w in words ] # words_ is to Distinguish the variable from pythpn keywords

    mask = np.array(mask)                                          ## convert the mask to array to be used in fancy indexing 

    data = tfidf_vectorizer_vectors[:,mask].toarray() # converting the matrix to array using masking + slicing 

    # create data frame with N column (the number of most frequent words) and we will match the index with the df_modified_1
    # which contain the other feature to be merger to construct the final dataframe which will be the input to the model

    return pd.DataFrame(data=data,columns=[f"word_{word}" for word in words_], index= dataframe.index) 

