import os
import shutil
import sys


# set new trained AI models as active for new predictions
from termcolor import colored


def update_main():
    # Check if there are new trained AI's to update
    if len(os.listdir(f"{sys.path[0]}/ml_models/new_models/")) == 0:
        print(colored("No new trained ai found.\nFirst train new AI's", "yellow"))
        return
    # Move active to old
    if len(os.listdir(f"{sys.path[0]}/ml_models/active_models/")) != 0:
        update_models(f"{sys.path[0]}/ml_models/active_models/", f"{sys.path[0]}/ml_models/old_models/")
    # New to Active
    update_models(f"{sys.path[0]}/ml_models/new_models/", f"{sys.path[0]}/ml_models/active_models/")


def restore():
    # Old to active
    if len(os.listdir(f"{sys.path[0]}/ml_models/old_models/")) != 0:
        update_models(f"{sys.path[0]}/ml_models/old_models/", f"{sys.path[0]}/ml_models/active_models/")
    else:
        print(colored("No old AI's to restore.", "yellow"))
        return


def update_models(source_dir, target_dir):
    for file_name in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

