import pandas as pd
import numpy as np
import math
import src.utils as utils
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from torch.utils.data import DataLoader
from src.model.dnn import DNN
import torch
from sklearn.metrics import top_k_accuracy_score
import recmetrics

log = get_logger(logger_name= "Model comparison", save_file= False)


def main():
    articles_df = pd.read_csv('./data/shared_articles.csv')
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    df = pd.read_csv('./data/users_interactions.csv')
    event_type_strength = {
        'VIEW': 0,
        'LIKE': 1, 
        'BOOKMARK': 2, 
        'FOLLOW': 3,
        'COMMENT CREATED': 4,  
    }

    num_dnn_output = df.eventType.nunique()

    df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])

    catagory_variables = ['contentId', 'personId']
    continuous_variables = []
    df, mapping_dict = utils.data_preprocessing(df, catagory_variables)
    variables = catagory_variables + continuous_variables + ['year', 'month', 'dow', 'hour']
    log.info('Data preprocessed !')

    train_dataset, val_dataset = train_test_split(df, test_size= 0.2, stratify= df["eventStrength"])
    val_dataset, test_dataset = train_test_split(val_dataset, test_size= 0.5, stratify= val_dataset["eventStrength"])
    log.info("split dataset completed !")

    train_dict = utils.convert_to_data_dict(train_dataset, variables, continuous_variables)
    val_dict = utils.convert_to_data_dict(val_dataset, variables, continuous_variables)
    test_dict = utils.convert_to_data_dict(test_dataset, variables, continuous_variables)


    test = test_dataset.copy().groupby('personId', as_index= False)['contentId'].agg({'actual': (lambda x: list(set(x)))})
    test = test.set_index("personId")

    # unique_person_id_test = test_dataset['personId'].unique()
    # unique_person_id_train = train_dataset['personId'].unique()
    # test_uid_to_contentIDs_dict = {uid: test_dataset[test_dataset['personId'] == int(uid)].contentId.values for uid in unique_person_id_test}
    # train_uid_to_contentIDs_dict = {uid: train_dataset[train_dataset['personId'] == int(uid)].contentId.values for uid in unique_person_id_train}

    # del unique_person_id_test, unique_person_id_train

    train_dataset = utils.DatasetConverter(train_dict, variables, labels=torch.LongTensor(train_dataset['eventStrength'].tolist()).unsqueeze(1))
    val_dataset = utils.DatasetConverter(val_dict, variables, labels=torch.LongTensor(val_dataset['eventStrength'].tolist()).unsqueeze(1))
    test_dataset = utils.DatasetConverter(test_dict, variables, labels=torch.LongTensor(test_dataset['eventStrength'].tolist()).unsqueeze(1))
    log.info("Convert to torch dataset completed !")

    train_loader = DataLoader(train_dataset, batch_size=2**13, shuffle=True, num_workers= 4)
    val_loader = DataLoader(val_dataset, batch_size=2**13, shuffle=True, num_workers= 4)
    test_loader = DataLoader(test_dataset, batch_size=2**13, shuffle=True, num_workers= 4)
    log.info("Convert to torch data loader completed !")

    model = DNN(mapping_dict, continuous_variables, catagory_variables, num_dnn_output)
    trained_model = utils.training(model, train_loader, val_loader, 'test3', epoch= 30)

    metrics_result = utils.calculate_metrics(trained_model, test_loader, test)


if __name__ == "__main__":
   main()






