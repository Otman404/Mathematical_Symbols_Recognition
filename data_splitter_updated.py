import math
import os
import random
import shutil
from collections import defaultdict
from os.path import abspath, join, basename, exists


def get_containing_folder_name(path):
    # dirname has inconsistent behavior when path has a trailing slash
    full_containing_path = abspath(join(path, os.pardir))
    return basename(full_containing_path)

"""
this code adapted from https://github.com/keras-team/keras/issues/5862#issuecomment-356121051
"""
def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct,
                                           stratify=True, seed=None):
    prev_state = None
    if seed:
        prev_state = random.getstate()
        random.seed(seed)

    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory", testing_data_dir)
    else:
        print(
            testing_data_dir, "not empty, did not remove contents")

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory", training_data_dir)
    else:
        print(training_data_dir, "not empty, did not remove contents")

    files_per_class = defaultdict(list)

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = basename(subdir)

        # Don't create a subdirectory for the root directories
        if category_name in map(basename, [all_data_dir, training_data_dir, testing_data_dir]):
            continue

        # filtered past top-level dirs, now we're in a category dir
        files_per_class[category_name].extend([join(abspath(subdir), file) for file in files])

    # keep track of train/validation split for each category
    split_per_category = defaultdict(lambda: defaultdict(int))
    # create train/validation directories for each class
    class_directories_by_type = defaultdict(lambda: defaultdict(str))
    for category in files_per_class.keys():
        training_data_category_dir = join(training_data_dir, category)
        if not exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)
        class_directories_by_type['train'][category] = training_data_category_dir

        testing_data_category_dir = join(testing_data_dir, category)
        if not exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)
        class_directories_by_type['validation'][category] = testing_data_category_dir

    if stratify:
        for category, files in files_per_class.items():

            random.shuffle(files)
            last_index = math.ceil(len(files) * testing_data_pct)
            # print('files upto index {} to val'.format(last_index))
            # print('category {} train/validation: {}/{}'.format(category, len(files[:last_index]),
            #                                                    len(files[last_index:])))
            for file in files[:last_index]:
                testing_data_category_dir = class_directories_by_type['validation'][category]
                # print('moving {} to {}'.format(file, join(testing_data_category_dir, basename(file))))
                shutil.move(file, join(testing_data_category_dir, basename(file)))
                split_per_category['validation'][category] += 1
            for file in files[last_index:]:
                training_data_category_dir = class_directories_by_type['train'][category]
                # print('moving {} to {}'.format(file, join(training_data_category_dir, basename(file))))
                shutil.move(file, join(training_data_category_dir, basename(file)))
                split_per_category['train'][category] += 1

    else:  # not stratified, move a fraction of all files to validation
        files = []
        for file_list in files_per_class.values():
            files.extend(file_list)

        random.shuffle(files)
        last_index = math.ceil(len(files) * testing_data_pct)
        for file in files[:last_index]:
            category = get_containing_folder_name(file)
            directory = class_directories_by_type['validation'][category]
            shutil.move(file, join(directory, basename(file)))
            split_per_category['validation'][category] += 1
        for file in files[last_index:]:
            category = get_containing_folder_name(file)
            directory = class_directories_by_type['train'][category]
            shutil.move(file, join(directory, basename(file)))
            split_per_category['train'][category] += 1

    if seed:
        random.setstate(prev_state)
    return split_per_category