import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from dev_model import get_MLP

model_MLP = None

MAX_SEQUENCE_LENGTH = 30

def init_model(modelname):
    global model_MLP
    if modelname == "MLP":
        model_MLP = get_MLP()
        model_MLP.load_weights("./model/MLP_no_feature_engineering_weights.h5")
        print("model " + modelname + " is loaded")
    else:
        raise Exception("no such model equipped")


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def get_prediction(q1_str, q2_str):
    global model_MLP
    print("checking "+ q1_str + " and " + q2_str)
    q1_words = text_to_wordlist(q1_str)
    q2_words = text_to_wordlist(q2_str)
    with open("./model/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle) 
    q1_seq = tokenizer.texts_to_sequences([q1_words])
    q2_seq = tokenizer.texts_to_sequences([q2_words])
    q1_seq_pad = pad_sequences(q1_seq, maxlen=MAX_SEQUENCE_LENGTH)
    q2_seq_pad = pad_sequences(q2_seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred_result = model_MLP.predict((q1_seq_pad, q2_seq_pad))
    pred_result += model_MLP.predict((q2_seq_pad, q1_seq_pad))
    pred_result /= 2
    prob = pred_result[0][0]
    if prob > 0.5:
        return "same meaning"
    else:
        return "not the same"
