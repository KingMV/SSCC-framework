import json
import os
import os.path as osp
import shutil


def load_json(path):
    with open(path) as data_file:
        param = json.load(data_file)
    return param


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def copy_file(src_file, dst_dir):
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copy(src_file, dst_dir)


def save_running_args(to_saved_args, path):
    with open(path, 'a+') as f:
        for k, v in to_saved_args.items():
            f.writelines('{0}: {1} \n'.format(k, v))


def store_current_training_args(log_dir, saving_name, to_saved_args):
    os.makedirs(log_dir, exist_ok=True)
    saving_path = osp.join(log_dir, saving_name)
    save_running_args(to_saved_args, saving_path)

    # copy the dataset config file to the log dir
    if to_saved_args.dataset is not None:
        t = to_saved_args.dataset
        src_file_path = osp.join('ccm/dataset', '{}/{}_Config.py'.format(t, t))
        copy_file(src_file_path, log_dir)

    # copy the trainer file to the log dir
    if to_saved_args.method is not None:
        t = to_saved_args.method
        src_file_path = osp.join('ccm/cores/', '{}.py'.format(t))
        copy_file(src_file_path, log_dir)
