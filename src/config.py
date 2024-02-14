import os

Config = {}


PACKAGE_DIR = os.path.dirname(os.path.abspath("__file__"))
_data_path = 'raw_data/avazu-ctr-prediction/'


Config['log_dir'] = os.path.join(PACKAGE_DIR, 'logs')


Config['data'] = {
    "train_path":  os.path.join(os.path.join(PACKAGE_DIR, _data_path), 'train.gz'),
    "test_path":  os.path.join(os.path.join(PACKAGE_DIR, _data_path), 'test.gz'),
}


Config['training_setting'] = {
    'num_threads': int(os.getenv("TRAINING_THREADS", 6)), 
    'num_workers': int(os.getenv("TRAINING_NUM_WORKERS", 10))
    }
