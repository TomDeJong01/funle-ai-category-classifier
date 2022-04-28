from db import db_controller
import json
import sys
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn import metrics


def train_main():
    # load categorised assignments and split in training and validation data
    dataset = db_controller.get_categorised_assignments()
    X_data, y_data = split_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=4)
    y_train_dnn, y_test_dnn = y_train - 1, y_test - 1
    X_train_dnn, X_test_dnn = tfidf_vectorizer(X_train, X_test)

    # Create new AI's
    svm = build_svm()
    rf = build_rf()
    gb = build_gb()
    dnn = build_dnn(X_train_dnn.shape[1])

    # Train AIs
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    dnn.fit(X_train_dnn, y_train_dnn, validation_data=(X_test_dnn, y_test_dnn), epochs=30, batch_size=30, verbose=2)

    # Save performance of AI's
    save_performance_scores(svm.predict_proba(X_test),
                            rf.predict_proba(X_test),
                            gb.predict_proba(X_test),
                            dnn.predict(X_test_dnn),
                            y_test)

    # Save trained AI objects
    save_classifier(svm, "svm.pkl")
    save_classifier(rf, "rf.pkl")
    save_classifier(gb, "gb.pkl")
    save_dnn_model(dnn)


def split_dataset(dataset):
    X_data = np.array([]).astype(str)
    y_data = np.array([]).astype(int)
    for row in dataset:
        X_data = np.append(X_data, row["Title"].lower())
        y_data = np.append(y_data, row["CategoryId"])
    return X_data, y_data


def get_accuracy_scores(proba, y_test):
    accuracy = metrics.accuracy_score(y_test, categoryids_from_probabilities(proba))
    y_pred_over_threshold = np.array([])
    y_true_over_threshold = np.array([])
    predictions_under_threshold = 0

    for i in range(len(y_test)):
        if proba[i].max() >= 0.8:
            y_pred_over_threshold = np.append(y_pred_over_threshold, np.where(proba[i] == proba[i].max())[0][0] + 1)
            y_true_over_threshold = np.append(y_true_over_threshold, y_test[i])
        else:
            predictions_under_threshold += 1
    accuracy_over_threshold = metrics.accuracy_score(y_true_over_threshold, y_pred_over_threshold)
    return round(accuracy, 3), round(accuracy_over_threshold, 3), predictions_under_threshold


def build_svm():
    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', CalibratedClassifierCV(LinearSVC())),
                     ])


def build_rf():
    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=300)),
                     ])


def build_gb():
    return Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', GradientBoostingClassifier(n_estimators=150)),
                     ])


def build_dnn(shape):
    # shape = word vocabulary size (corpus)
    # Dense = layer in Neural network
    model = Sequential()  # initialize neural network
    # node = 512  # number of nodes
    node = int(round(shape*0.8))  # number of nodes
    n_layers = 6  # number of  hidden layer
    n_classes = 7  # aantal categorien
    dropout = 0.5

    model.add(Dense(node, input_dim=shape, activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0, n_layers):
        model.add(Dense(node, input_dim=node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# Convert text into a readable shape for neural network.
def tfidf_vectorizer(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = np.array(vectorizer.fit_transform(X_train).toarray())
    X_test = np.array(vectorizer.transform(X_test).toarray())
    with open(f"{sys.path[0]}/ml_models/new_models/vectorizer.pkl", 'wb') as file:
        pickle.dump(vectorizer, file)
    return X_train, X_test


def categoryids_from_probabilities(probabilities):
    return np.argmax(probabilities, axis=1) + 1


def save_performance_scores(svm_proba, rf_proba, gb_proba, dnn_proba, y_test):
    svm_acc, svm_acc_ot, svm_predictions_under_threshold = get_accuracy_scores(svm_proba, y_test)
    rf_acc, rf_acc_ot, rf_predictions_under_threshold = get_accuracy_scores(rf_proba, y_test)
    gb_acc, gb_acc_ot, gb_predictions_under_threshold = get_accuracy_scores(gb_proba, y_test)
    dnn_acc, dnn_acc_ot, dnn_predictions_under_threshold = get_accuracy_scores(dnn_proba, y_test)

    score_data = {
        "total_tests": len(y_test),
        "svm_accuracy": svm_acc,
        "svm accuracy high probability": svm_acc_ot,
        "svm tests discared due to low probability": svm_predictions_under_threshold,
        "rf_accuracy": rf_acc,
        "rf accuracy high probability": rf_acc_ot,
        "rf tests discared due to low probability": rf_predictions_under_threshold,
        "gb_accuracy": gb_acc,
        "gb accuracy high probability": gb_acc_ot,
        "gb tests discared due to low probability": gb_predictions_under_threshold,
        "dnn_accuracy": dnn_acc,
        "dnn accuracy high probability": dnn_acc_ot,
        "dnn tests discared due to low probability": dnn_predictions_under_threshold
    }
    print(score_data)

    with open(f"{sys.path[0]}/ml_models/new_models/performance.json", 'w', encoding='utf-8') as file:
        json.dump(score_data, file, ensure_ascii=False, indent=4)


def save_classifier(model, name):
    with open(f"{sys.path[0]}/ml_models/new_models/{name}", 'wb') as file:
        pickle.dump(model, file)


def save_dnn_model(dnn):
    dnn.save(f"{sys.path[0]}/ml_models/new_models/dnn.h5")
