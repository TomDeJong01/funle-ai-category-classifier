import numpy as np
from db import db_connector


def get_categorised_assignments():
    db = db_connector.DbConnector()
    dataset = db.get_categorised_assignments()
    db.close()
    return dataset


def get_uncategorised_assignments():
    db = db_connector.DbConnector()
    dataset = db.get_uncategorised_assignments()
    db.close()
    return dataset


def save_prediction(classifier, predictions):
    db = db_connector.DbConnector()
    for prediction in predictions:
        db.save_prediction(classifier, prediction)
        db.connection.commit()
    db.close()


def update_assignment_category_ids(prediction_results):
    db = db_connector.DbConnector()
    for prediction in prediction_results:
        db.update_category(prediction)
        db.connection.commit()
    db.close()

# SQL create category_prediction
# create table category_prediction(
#     "AssignmentId"        integer,
#     "PredictedCategoryId" integer,
#     "Probability"         numeric,
#     "Classifier"          varchar,
#     constraint category_prediction_pk
#         unique ("AssignmentId", "Classifier"));
