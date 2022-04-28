import json
import sys


# Load and print performance files of active and newly trained AI's
def compare_main():
    try:
        print("new")
        with open(f"{sys.path[0]}/ml_models/new_models/performance.json") as json_file:
            data = json.load(json_file)
            print(json.dumps(data, indent=4))
    except FileNotFoundError:
        print(f"No performance file found for new AI.\nTrain a new AI with -t\nOnly showing performance of active AI")

    try:
        with open(f"{sys.path[0]}/ml_models/active_models/performance.json") as json_file:
            data = json.load(json_file)
            print(json.dumps(data, indent=4))
    except FileNotFoundError:
        print(f"No performance file found, if application isnt working train a new AI and update")
