# from db import db_controller
# import json
# import sys
# import numpy as np
# import pickle
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from tensorflow import keras
# from keras.layers import Dropout, Dense
# from keras.models import Sequential
# from sklearn import metrics
# import time
# from kerastuner.tuners import RandomSearch
# from kerastuner.engine.hyperparameters import HyperParameters
#
#
# def split_dataset(dataset):
#     X_data = np.array([]).astype(str)
#     y_data = np.array([]).astype(int)
#     for row in dataset:
#         X_data = np.append(X_data, row["Title"].lower())
#         y_data = np.append(y_data, row["CategoryId"])
#     return X_data, y_data
#
#
# def tfidf_vectorizer(X_train, X_test, MAX_NB_WORDS=75000):
#     vectorizer = TfidfVectorizer(max_features=MAX_NB_WORDS)
#     X_train = np.array(vectorizer.fit_transform(X_train).toarray())
#     X_test = np.array(vectorizer.transform(X_test).toarray())
#     return X_train, X_test
#
#
#
# LOG_TIME = f"{int(time.time())}"
# dataset = db_controller.get_categorised_assignments()
# X_data, y_data = split_dataset(dataset)
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
# y_train_dnn, y_test_dnn = y_train - 1, y_test - 1
# X_train_dnn, X_test_dnn = tfidf_vectorizer(X_train, X_test)
#
# def build_dnn(hp):
#     model = Sequential()  # initialize neural network
#     nClasses = 7
#     node = hp.Int("nodes", min_value=int(round(X_train_dnn.shape[1]*0.5)),
#                   max_value=int(X_train_dnn.shape[1]), step=int(round(X_train_dnn.shape[1]*0.1)))  # number of nodes
#     # node = round(X_train_dnn.shape[1]*0.8)  # number of nodes
#     nLayers = hp.Int("nLayers", 4, 8, 1)  # number of  hidden layer
#     # dropout = hp.Float("dropout", 0.5, 0.8, 1)
#     dropout = 0.5
#
#     model.add(Dense(node, input_dim=X_train_dnn.shape[1], activation='relu'))
#     model.add(Dropout(dropout))
#     new_node = node
#     for i in range(0, nLayers):
#         input_dim = node
#         new_node = int(round(new_node*0.8))
#         model.add(Dense(new_node, input_dim=input_dim, activation='relu'))
#         model.add(Dropout(dropout))
#
#         # model.add(Dense(node, input_dim=new_node, activation='relu'))
#         # model.add(Dropout(dropout))
#     model.add(Dense(nClasses, activation='softmax'))
#
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model
#
#
# print(X_train_dnn.shape[1])
# tuner = RandomSearch(
#     build_dnn,
#     objective="val_accuracy",
#     max_trials=20,
#     executions_per_trial=2,
#     directory=f"{sys.path[0]}/ml_models/tune"
# )
# tuner.search(
#     x=X_train_dnn,
#     y=y_train_dnn,
#     epochs=20,
#     batch_size=100,
#     validation_data=(X_test_dnn, y_test_dnn)
# )
# tuner.results_summary()
# # with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
# with open(f"{sys.path[0]}/ml_models/tune/{int(time.time())}.pkl",
#           "wb") as f:
#     pickle.dump(tuner, f)
#
#
# print(tuner.get_best_hyperparameters()[0].values)
# print(tuner.get_best_models()[0].summary())
#
# # best 1300 nodes, 4 Nlayesr
# # best  1341 nodes, 6 layers
# # 745, 6
