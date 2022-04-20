import psycopg2
import psycopg2.extras
from dotenv import dotenv_values, find_dotenv


class DbConnector:
    def __init__(self):
        self.ENV = dotenv_values(find_dotenv("../.env"))
        self.assignments_table = self.ENV["ASSIGNMENTS_TABLE"]
        self.conn = self.connect()
        self._cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def connect(self):
        try:
            return psycopg2.connect(
                host=self.ENV["HOST"],
                database=self.ENV["DATABASE"],
                user=self.ENV["USER"],
                port=self.ENV["PORT"],
                password=self.ENV["PASSWORD"])
        except psycopg2.Error as err:
            print(f"connection error: {err.__class__.__name__}")
            pass

    @property
    def connection(self):
        return self.conn

    @property
    def cursor(self):
        return self._cursor

    def fetchall(self):
        return self.cursor.fetchall()

    def close(self):
        self.connection.close()

    def get_uncategorised_assignments(self):
        self.cursor.execute(f"""SELECT "Id", "Title" 
                            FROM {self.assignments_table} 
                            WHERE "CategoryId" ISNULL
                            AND "Title" NOTNULL;""")
        return self.fetchall()

    def get_categorised_assignments(self):
        self.cursor.execute(f"""SELECT "Id", "Title", "CategoryId"
                            FROM {self.assignments_table}
                            WHERE "CategoryId" NOTNULL;""")
        return self.fetchall()

    def save_prediction(self, classifier, prediction):
        self.cursor.execute(f"""INSERT INTO "category_prediction"
            ("AssignmentId", "Classifier", "PredictedCategoryId", "Probability")
            VALUES({prediction["Id"]}, 
            '{classifier}', 
            {prediction["PredictedCategoryId"]}, 
            {prediction["PredictionProbability"]})
            ON CONFLICT ("AssignmentId", "Classifier") DO 
            UPDATE SET "PredictedCategoryId" = {prediction["PredictedCategoryId"]},
            "Probability" = {prediction["PredictionProbability"]}; 
            """)

    def update_category(self, prediction):
        self.cursor.execute(f"""UPDATE {self.assignments_table} 
            SET  "CategoryId" = {prediction["PredictedCategoryId"]} 
            SET  "PredictedCategoryId" = {prediction["PredictedCategoryId"]} 
            WHERE "Id" = {prediction["Id"]}; """)



