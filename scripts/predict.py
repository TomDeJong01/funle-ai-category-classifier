import numpy as np
from tensorflow import keras
from db import db_controller
import sys
import pickle

def predict_main():
    dataset = db_controller.get_uncategorised_assignments()
    titles, ids = split_dataset(dataset)
    svm = load_model("svm.pkl")
    rf = load_model("rf.pkl")
    gb = load_model("gb.pkl")
    dnn = load_dnn()

    vectorizer = load_vectorizer()
    titles_tfidf = np.array(vectorizer.transform(titles).toarray())

    svm_predicted = create_prediction_result_object(ids, svm.predict_proba(titles), titles)
    rf_predicted = create_prediction_result_object(ids, rf.predict_proba(titles), titles)
    gb_predicted = create_prediction_result_object(ids, gb.predict_proba(titles), titles)
    dnn_predicted = create_prediction_result_object(ids, dnn.predict(titles_tfidf), titles)

    db_controller.save_prediction("svm", svm_predicted)
    db_controller.save_prediction("rf", rf_predicted)
    db_controller.save_prediction("gb", gb_predicted)
    db_controller.save_prediction("dnn", dnn_predicted)

    pool_predictions(svm_predicted, rf_predicted, gb_predicted, dnn_predicted)


def split_dataset(dataset):
    titles = np.array([]).astype(str)
    ids = np.array([]).astype(int)
    for row in dataset:
        titles = np.append(titles, row["Title"].lower())
        ids = np.append(ids, row["Id"])
    return titles, ids


def load_vectorizer():
    with open(f"{sys.path[0]}/ml_models/active_models/vectorizer.pkl", 'rb') as file:
        return pickle.load(file)


def load_dnn():
    return keras.models.load_model(f"{sys.path[0]}/ml_models/active_models/dnn.h5")


def load_model(name):
    file_path = f"{sys.path[0]}/ml_models/active_models/{name}"
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
    pooled_results = np.array([])
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
            categoryId = None
            if np.average(proba) >= 0.8:
                categoryId = dnn_predicted[i]["PredictedCategoryId"]
            pooled_results = np.append(pooled_results, {
                "Id": dnn_predicted[i]["Id"],
                "CategoryId": categoryId,
                "PredictedCategoryId": gb_predicted[i]["PredictedCategoryId"]
            })
    print("update categories*")
    # db_controller.update_assignment_category_ids(pooled_results)


