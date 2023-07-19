from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from keras import backend as K
import pandas as pd
from keras.layers import LSTM, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GRU
from keras.layers.embeddings import Embedding
from keras.models import Model
import re
from keras.preprocessing.text import Tokenizer
from pyarabic import araby
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras
from pytorch_pretrained_bert.modeling import BertModel
from tensorflow.keras.models import Sequential,load_model

# Functions for preprocessing dataset
from tensorflow.python.keras.layers import Bidirectional


def clean(text):
    text = text.replace("<br/>", " ")
    strip_special_chars = re.compile(u'[^\u0621-\u064a ]')
    return re.sub(strip_special_chars, " ", text)


def process(text):
    text = araby.strip_tashkeel(text)  # delete *tashkil
    text = re.sub('\ـ+', ' ', text)  # delete letter madda
    text = re.sub('\ر+', 'ر', text)  # duplicate ra2
    text = re.sub('\اا+', 'ا', text)  # duplicate alif
    text = re.sub('\ووو+', 'و', text)  # duplicate waw (more than 3 times goes to 1
    text = re.sub('\ههه+', 'ههه', text)  # duplicate ha2 (more than 3 times goes to 1
    text = re.sub('\ةة+', 'ة', text)
    text = re.sub('\ييي+', 'ي', text)
    text = re.sub('أ', 'ا', text)  # after to avoid mixing
    text = re.sub('آ', 'ا', text)  # after to avoid mixing
    text = re.sub('إ', 'ا', text)  # after to avoid mixing
    text = re.sub('ة', 'ه', text)  # after ةة to avoid mixing ههه
    text = re.sub('ى', 'ي', text)
    text = " ".join(text.split())  # delete multispace
    return text

data = pd.read_csv('dataset.csv')
data = data[:39560]
data['clean_text'] = data['text'].apply(lambda cw: clean(cw))
data['clean_text_process'] = data['clean_text'].apply(lambda cw: process(cw))
print(data['clean_text_process'])

text = data['clean_text_process']
print(text)
text_list = []
for i in range(len(text)):
    text_list.append(text[i])

sentiment = data['sentiment']
counter = data.groupby(data['sentiment'])['sentiment'].count()
print("Number of positive and negative label: ",counter)
X =  np.array(list(text_list))
y = np.array(list(map(lambda x: 1 if x == "Positive" else 0, sentiment)))
print("Shape of X and y :",X.shape, " ", y.shape)

print("Before Over Sampling, count of the label '1': {}".format(sum(y == 1)))
print("Before Over Sampling, count of the label '0': {} \n".format(sum(y == 0)))
from imblearn.over_sampling import SMOTE
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
sm1 = SMOTE(random_state = 2)


maxLen = 150
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_train_indices = tokenizer.texts_to_sequences(X)
X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)

X__res, y__res = sm1.fit_resample(X_train_indices, y)
print('After Over Sampling, the shape of the X: {}'.format(X__res.shape))
print('After Over Sampling, the shape of the y: {} \n'.format(y__res.shape))
print("After Over Sampling, count of the label '1': {}".format(sum(y__res == 1)))
print("After Over Sampling, count of the label '0': {}".format(sum(y__res == 0)))


X_train_indices, X_test_indices, Y_train, Y_test = train_test_split(X__res, y__res, test_size=0.2, random_state=45)

BERT_FP = 'DarijaBERT/'

def get_bert_embed_matrix():
    bert = BertModel.from_pretrained(BERT_FP)
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat

embedding_matrix = get_bert_embed_matrix()
embedding_layer = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)

# Define functions for each base-classier
def CNN(input_shape):
    X_indices = Input(input_shape)
    embeddings = embedding_layer(X_indices)
    X = Conv1D(512, 3, activation='relu')(embeddings)
    X = MaxPooling1D(3)(X)
    # X = Conv1D(256, 3, activation='relu')(X)
    # X = MaxPooling1D(3)(X)
    X = Conv1D(256, kernel_size=1, activation='relu')(X)
    X = Dropout(0.8)(X)
    X = MaxPooling1D(3)(X)
    X = GlobalMaxPooling1D()(X)
    X = Dense(256, activation='relu')(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=X_indices, outputs=X)
    return model

def LSTM_(input_shape):
    X_indices = Input(input_shape)
    embeddings = embedding_layer(X_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.6)(X)
    X = LSTM(128)(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=X_indices, outputs=X)
    return model

def BiLSTM(input_shape):
    X_indices = Input(input_shape)
    embeddings = embedding_layer(X_indices)
    X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
    X = Dropout(0.6)(X)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=X_indices, outputs=X)
    return model

def GRU_(input_shape):
    X_indices = Input(input_shape)
    embeddings = embedding_layer(X_indices)
    X = GRU(128, return_sequences=True)(embeddings)
    X = Dropout(0.6)(X)
    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.6)(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=X_indices, outputs=X)
    return model
maxLen = 150

# Fit the base classifiers 
adam = keras.optimizers.Adam(learning_rate=0.0001)
model1 = CNN((maxLen,))
print(model1.summary())
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.fit(X_train_indices, Y_train, batch_size=64, epochs=1)
model1.save('model1.h5')
print("model 1 CNN saved")

model2 = LSTM_((maxLen,))
print(model2.summary())
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X_train_indices, Y_train, batch_size=64, epochs=1)
model2.save('model2.h5')
print("model 2 LSTM saved")
#
model3 = BiLSTM((maxLen,))
print(model3.summary())
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model3.fit(X_train_indices, Y_train, batch_size=64, epochs=1)
model3.save('model3.h5')
print("model 3 BiLSTM saved")

model4 = GRU_((maxLen,))
print(model4.summary())
model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model4.fit(X_train_indices, Y_train, batch_size=64, epochs=1)
model4.save('model4.h5')
print("model 4 GRU saved")


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

dependencies = {
    'f1_m': f1_m
}

# load models from file
def load_all_models():
  all_models = list()
  model1 = load_model('model1.h5', custom_objects=dependencies)
  model2 = load_model('model2.h5', custom_objects=dependencies)
  model3 = load_model('model3.h5', custom_objects=dependencies)
  model4 = load_model('model4.h5', custom_objects=dependencies)
  all_models.append(model1)
  all_models.append(model2)
  all_models.append(model3)
  all_models.append(model4)
  print('>loaded models ')
  return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    preds1 = members[0].predict(inputX)
    preds2 = members[1].predict(inputX)
    preds1_ = preds1.reshape((preds1.shape[0], preds1.shape[1], 1))
    preds2_ = preds2.reshape((preds2.shape[0], preds2.shape[1], 1))
    preds3 = members[2].predict(inputX)
    preds4 = members[3].predict(inputX)
    stackX = np.column_stack((preds1_, preds2_, preds3, preds4))
    print('stackX shape: ', stackX.shape)
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], (stackX.shape[1]*stackX.shape[2])))
    return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = LogisticRegression(solver='lbfgs', max_iter=100) #meta learner
    model.fit(stackedX, inputy)
    return model

members = load_all_models()
print('Loaded the models')

modelStacked = fit_stacked_model(members, X_train_indices, Y_train)
print('stacked model fited')

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

# evaluate proposed model on test set
yhat = stacked_prediction(members, modelStacked, X_test_indices)
score = cross_val_score(modelStacked,X_test_indices,Y_test,cv = 5,scoring = 'accuracy')
print("The accuracy score of stacked model is:",score.mean())

#evaluate single classifiers (CNN, LSTM, BiLSTM, GRU)
i = 0
for model in members:
    i+=1
    model.predict(X_test_indices)
    print('evaluation of model {} is '.format(i),model.evaluate(X_test_indices, Y_test))
