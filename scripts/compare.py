import json
import sys
from termcolor import colored


# Load and print performance files of active and newly trained AI's
def compare_main():
    try:
        # print(colored("Hello", "red"), colored("world", "yellow"))
        with open(f"{sys.path[0]}/ml_models/new_models/performance.json") as json_file:
            data = json.load(json_file)
            print(json.dumps(data, indent=4))
    except FileNotFoundError:
        print(colored("No performance file found for new AI.\nTrain a new AI with -t\n"
                      "Only showing performance of active AI", "yellow"))

    try:
        with open(f"{sys.path[0]}/ml_models/active_models/performance.json") as json_file:
            data = json.load(json_file)
            print(json.dumps(data, indent=4))
    except FileNotFoundError:
        print(colored("No performance file found for active models!"
                      "If application isn't working train a new AI and update", "red"))
