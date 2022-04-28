import os
import shutil
import sys


# set new trained AI models as active for new predictions
def update_main():
    source_dir = f"{sys.path[0]}/ml_models/new_models/"
    target_dir = f"{sys.path[0]}/ml_models/active_models/"
    file_names = os.listdir(source_dir)

    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))
