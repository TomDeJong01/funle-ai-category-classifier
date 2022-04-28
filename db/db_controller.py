from db import db_connector


def get_categorised_assignments():
    db = db_connector.DbConnector()
    dataset = None
    try:
        dataset = db.get_categorised_assignments()
    except Exception as e:
        print(e)
    finally:
        db.close()
    return dataset


def get_uncategorised_assignments():
    db = db_connector.DbConnector()
    dataset = None
    try:
        dataset = db.get_uncategorised_assignments()
    except Exception as e:
        print(e)
    finally:
        db.close()
    return dataset


def save_prediction(classifier, predictions):
    db = db_connector.DbConnector()
    try:
        for prediction in predictions:
            db.save_prediction(classifier, prediction)
            db.connection.commit()
    except Exception as e:
        print(e)
    finally:
        db.close()


def update_assignment_category_ids(prediction_results):
    db = db_connector.DbConnector()
    try:
        for prediction in prediction_results:
            db.update_category(prediction)
            db.connection.commit()
    except Exception as e:
        print(e)
    finally:
        db.close()
