import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras

from db import db_connector, db_controller
import sys
import pickle

from db.db_controller import save_prediction


def predict_main():
    dataset = db_controller.get_uncategorised_assignments()
    titles, ids = split_dataset(dataset)
    svm = load_model("svm.pkl")
    rf = load_model("rf.pkl")
    gb = load_model("gb.pkl")
    dnn = load_dnn()

    vectorizer = load_vectorizer()
    titles_tfidf =  np.array(vectorizer.transform(titles).toarray())

    svm_predicted = create_prediction_result_object(ids, svm.predict_proba(titles), titles)
    rf_predicted = create_prediction_result_object(ids, rf.predict_proba(titles), titles)
    gb_predicted = create_prediction_result_object(ids, gb.predict_proba(titles), titles)
    dnn_predicted = create_prediction_result_object(ids, dnn.predict(titles_tfidf), titles)

    # save_prediction("svm", svm_predicted)
    # save_prediction("rf", rf_predicted)
    # save_prediction("gb", gb_predicted)
    # save_prediction("dnn", dnn_predicted)

    pool_predictions(svm_predicted, rf_predicted, gb_predicted, dnn_predicted)


def split_dataset(dataset):
    titles = np.array([]).astype(str)
    ids = np.array([]).astype(int)
    for row in dataset:
        titles = np.append(titles, row["Title"].lower())
        ids = np.append(ids, row["Id"])
    return titles, ids


def load_vectorizer():
    with open(f"{sys.path[0]}/ml_models/new_models/vectorizer", 'rb') as file:
        return pickle.load(file)


def load_dnn():
    return keras.models.load_model(f"{sys.path[0]}/ml_models/active_models/dnn.h5")


def load_model(name):
    file_path = f"{sys.path[0]}/ml_models/active_models/{name}"
    print(f"path: {file_path}")
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"{name} file not found")


def create_prediction_result_object(ids, predicted_probas, titles):
    prediction_results = np.array([])
    predicted_categories = np.argmax(predicted_probas, axis=1) + 1
    for i in range(0, len(ids)):
        prediction_results = np.append(prediction_results, {
            "Id": ids[i],
            "PredictedCategoryId": predicted_categories[i],
            "PredictionProbability": round(float(predicted_probas[i].max()), 3),
            "Title": titles[i]
        })
    return prediction_results


def pool_predictions(svm_predicted, rf_predicted, gb_predicted, dnn_predicted):
    any_under = 0
    avg_under = 0
    count = 0
    for i in range(len(svm_predicted)):
        predicted = np.array([svm_predicted[i]["PredictedCategoryId"],
                              rf_predicted[i]["PredictedCategoryId"],
                              gb_predicted[i]["PredictedCategoryId"],
                              dnn_predicted[i]["PredictedCategoryId"]])

        proba = np.array([svm_predicted[i]["PredictionProbability"],
                          rf_predicted[i]["PredictionProbability"],
                          gb_predicted[i]["PredictionProbability"],
                          dnn_predicted[i]["PredictionProbability"]])
        unique, counts = np.unique(predicted, return_counts=True)

        if len(unique) == 1:
            count +=1
            if np.amin(proba) < 0.8:
                # print(np.average(proba))
                any_under += 1
            if np.average(proba) < 0.8:
                print(f"{round(np.average(proba), 3)}, {proba}")
                avg_under += 1

    print(f"total:{len(svm_predicted)}count:{count}, min_under: {any_under}, avg_under: {avg_under}")
        # print(len(unique))
        # print(f"u: {unique}: {counts}")
    # print(f"len: {len(svm_predicted)}\n1: {count_1}\n2: {count_2}\n3: {count_3}\n4: {count_more}")

