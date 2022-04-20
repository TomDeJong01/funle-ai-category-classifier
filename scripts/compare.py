import json
import sys


def compare_main():
    with open(f"{sys.path[0]}/ml_models/new_models/performance.json") as json_file:
        data = json.load(json_file)
        print(json.dumps(data, indent=4))
    with open(f"{sys.path[0]}/ml_models/active_models/performance.json") as json_file:
        data = json.load(json_file)
        print(json.dumps(data, indent=4))




