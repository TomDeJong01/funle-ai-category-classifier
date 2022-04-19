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

