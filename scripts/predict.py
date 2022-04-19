import numpy as np
from db import db_connector, db_controller
import sys
import pickle


def predict_main():
    dataset = db_controller.get_uncategorised_assignments()
    titles, ids = split_dataset(dataset)
    svm = load_model("svm.pkl")
    rf = load_model("rf.pkl")
    gb = load_model("gb.pkl")
    dnn = load_model("dnn.h5")

    countvectorizer = svm["vect"]
    corpus = svm["vect"].get_feature_names_out()
    tokenizer = svm["vect"].build_tokenizer()

    svm_predicted_probas = svm.predict_proba(titles)
    rf_predicted_probas = rf.predict_proba(titles)
    gb_predicted_probas = gb.predict_proba(titles)

    svm_predicted = create_prediction_result_object(ids, svm_predicted_probas, titles)
    rf_predicted = create_prediction_result_object(ids, rf_predicted_probas, titles)
    gb_predicted = create_prediction_result_object(ids, gb_predicted_probas, titles)


    # evaluate_predictions(svm_predicted, rf_predicted, gb_predicted)
    test(svm_predicted, rf_predicted, gb_predicted, svm)


def split_dataset(dataset):
    titles = np.array([]).astype(str)
    ids = np.array([]).astype(int)
    for row in dataset:
        titles = np.append(titles, row["Title"].lower())
        ids = np.append(ids, row["Id"])
    return titles, ids


def load_model(name):
    file_path = f"{sys.path[0]}/ml_models/active_models/{name}"
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"{name} file not found")


def create_prediction_result_object(ids, predicted_probas, titles):
    prediction_results = np.array([])
    for i in range(0, len(ids)):
        predicted_category = np.where(predicted_probas[i] == predicted_probas[i].max())[0][0] + 1
        prediction_results = np.append(prediction_results, {
            "Id": ids[i],
            "PredictedCategoryId": predicted_category,
            "PredictionProbability": round(predicted_probas[i].max(), 3),
            "Title": titles[i]
        })
    return prediction_results


def test(svm_predicted, rf_predicted, gb_predicted, svm):
    countvectorizer = svm["vect"]
    corpus = countvectorizer.get_feature_names_out()
    tokenizer = countvectorizer.build_tokenizer()
    print(tokenizer)
    all_equal_count = 0
    two_cat_count = 0
    all_equal_low_prob = 0
    for i in range(0, len(svm_predicted)):
        predicted_categories = np.array([
            svm_predicted[i]["PredictedCategoryId"],
            rf_predicted[i]["PredictedCategoryId"],
            gb_predicted[i]["PredictedCategoryId"]])
        probas = [svm_predicted[i]["PredictionProbability"],
                  rf_predicted[i]["PredictionProbability"],
                  gb_predicted[i]["PredictionProbability"]]
        unique, counts = np.unique(predicted_categories, return_counts=True)
        category_result_counts = np.column_stack((unique, counts))
        if len(category_result_counts) == 1:
            all_equal_count += 1
            if any(j < 0.8 for j in probas):
                title = svm_predicted[i]["Title"]
                print(f"{title}")
                # print(f"{[countvectorizer.transform(title)]}, {title}")
                all_equal_low_prob +=1
        if len(category_result_counts) == 2:
            two_cat_count += 1
            # print("all results equal")
            # print(svm_predicted[i])

    print(f"all equal:{all_equal_count}/{len(svm_predicted)}\n"
          f"high prob:{all_equal_count - all_equal_low_prob}/{all_equal_count}")


def evaluate_predictions(svm_predicted, rf_predicted, gb_predicted):
    svm_over_80 = 0
    rf_over_80 = 0
    gb_over_80 = 0
    svm_uncertain = np.array([])
    rf_uncertain = np.array([])
    gb_uncertain = np.array([])

    svm_certain = np.array([])
    rf_certain = np.array([])
    gb_certain = np.array([])

    for i in range(0, len(svm_predicted)):
        if svm_predicted[i]["PredictionProbability"] >= 0.8:
            svm_over_80 += 1
            svm_certain = np.append(svm_certain, svm_predicted[i])
        else:
            svm_uncertain = np.append(svm_uncertain, svm_predicted[i])

        if rf_predicted[i]["PredictionProbability"] >= 0.8:
            rf_over_80 += 1
            rf_certain = np.append(rf_certain, rf_predicted[i])
        else:
            rf_uncertain = np.append(rf_uncertain, rf_predicted[i])

        if gb_predicted[i]["PredictionProbability"] >= 0.8:
            gb_over_80 += 1
            gb_certain = np.append(gb_certain, gb_predicted[i])
        else:
            gb_uncertain = np.append(gb_uncertain, gb_predicted[i])

    print(f"svm:\n{svm_uncertain[:100]}")
    print(f"rf:\n{rf_uncertain[:100]}")
    print(f"gb:\n{gb_uncertain[:100]}")
    print(f"total: {len(svm_predicted)}\nsvm: {svm_over_80}\nrf:  {rf_over_80}\n gb:  {gb_over_80}")
