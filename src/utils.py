import math
from sklearn import metrics
import torch
from torch.utils.data.dataset import Dataset
from typing import List, Union
import datetime
import numpy as np
import pandas as pd
from src.config import Config
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import top_k_accuracy_score


def convert_to_data_dict(data, variables, continuous_variables):
    return  {
            variables[i]: torch.FloatTensor(x) if variables[i] in continuous_variables else torch.LongTensor(x)
            for i, x in enumerate(zip(*data[variables].values))
        }

class DatasetConverter(Dataset):
    def __init__(self, inputs, variables, labels=None):
        self.variables = variables
        self.inputs = torch.stack(list(inputs.values()), dim=1)
        self.labels=labels

    def __getitem__(self, idx):
        if not isinstance(self.labels, type(None)):
            x = self.inputs[idx]
            return {'inputs': dict(zip(self.variables, x)), 'labels': self.labels[idx]}
        else:
            return {'inputs': dict(zip(self.variables, x))}

    def __len__(self):
        return len(self.inputs)

def data_preprocessing(df: pd.DataFrame, catagory_variables: List[str], mapping_dict:dict = None):
    timestamps = list(map(lambda timestamp: datetime.datetime.fromtimestamp(timestamp), df['timestamp']))

    df['hour'] = list(map(lambda timstamp: timstamp.hour, timestamps))
    df['dow'] = list(map(lambda timstamp: timstamp.weekday(), timestamps))
    df['year'] = list(map(lambda timstamp: timstamp.year, timestamps))
    df['month'] = list(map(lambda timstamp: timstamp.month, timestamps))

    catagory_variables += ['year', 'month', 'dow', 'hour']

    if not mapping_dict:
        mapping_dict = {}

        for cat in catagory_variables:
            keys = np.concatenate((np.unique(df[cat]), np.array(["other"])) , axis=0)

            value = [i for i, _ in enumerate(keys)]
            mapping_dict[cat]  = dict(zip(keys, value))


    for cat in catagory_variables:
        mapped_index = list(map(lambda value: 
                            mapping_dict[cat][str(value)]
                                if str(value) in mapping_dict[cat] 
                                    else mapping_dict[cat]['other']  , df[cat]))
        df[cat] = mapped_index
    return df, mapping_dict

def training(model, train_loader, val_loader, experient, lr:float = 0.001, epoch:int= 10, early_stop:bool = False):
    early_stopping = EarlyStopping(tolerance=2, min_delta=10)
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5)
    writer = SummaryWriter(f'run/{experient}')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.85, verbose=True)

    validationEpoch_loss = []
    for ep in range(epoch):
        step_loss = []
        y_true = []
        y_pred = []

        model.train()
        for iteration, batch_data in enumerate(train_loader):
            prediction = model(batch_data['inputs'])
            # loss = criterion(prediction, batch_data['labels'])
            loss = criterion(prediction, batch_data['labels'].reshape(-1))

            train_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss.append(train_loss)
            y_true.append(batch_data['labels'].detach().numpy())
            y_pred.append(prediction.detach().numpy())
            print (f'Epoch: [{ep+1}/{epoch}] | iteration: {iteration} | Loss: {np.array(step_loss).mean():.4f}')

        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight,  ep)
            writer.add_histogram(f'{name}.grad', weight.grad,  ep)
        y_true = np.concatenate(y_true).ravel()
        y_pred = np.concatenate(y_pred).ravel()
        # auc = metrics.roc_auc_score(y_true, y_pred)

        step_val_loss = []
        y_val_true = []
        y_val_pred = []

        for iteration, batch_data in enumerate(val_loader):
            with torch.inference_mode(): 
                prediction = model(batch_data['inputs'])
                loss = criterion(prediction, batch_data['labels'].reshape(-1)).item()

                step_val_loss.append(loss)
                y_val_true.append(batch_data['labels'].detach().numpy())
                y_val_pred.append(prediction.detach().numpy())

                if iteration % 10 == 0:
                    print (f'Epoch: [{ep+1}/{epoch}] | iteration: {iteration} | valLoss: {np.mean(step_val_loss):.4f}')

        y_val_true = np.concatenate(y_val_true).ravel()
        y_val_pred = np.concatenate(y_val_pred).ravel()
        validationEpoch_loss.append(np.mean(step_val_loss))
        # var_auc = metrics.roc_auc_score(y_val_true, y_val_pred)


        adjusted_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('lr', adjusted_lr, ep)
        scheduler.step()

        writer.add_scalars('loss', {'train': np.array(step_loss).mean(),
                                    'val':  np.array(step_val_loss).mean()}, ep)

        # writer.add_scalars('auc', {'train': auc,
        #                             'val':  var_auc}, ep)

        print("=" * 60)
        
        # early stopping
        if early_stop:
            if len(validationEpoch_loss) > 1:
                early_stopping(validationEpoch_loss[ep], validationEpoch_loss[ep-1])
                if early_stopping.early_stop:
                    print("We are at epoch:", ep)
                    break
    return model.eval()

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss, previous_validation_loss):
        if validation_loss > previous_validation_loss:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def compute_scores(query_embedding, item_embeddings, measure="DOT"):
    u = query_embedding
    V = item_embeddings
    if measure == "cosine":
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = np.dot(u, V.T)
    return scores

def calculate_metrics(trained_model, test_loader, test) -> dict:
    metrics_result = {}
    y_true = []
    y_prob = []
    with torch.inference_mode():
        for batch_data in test_loader:
            prediction = trained_model(batch_data['inputs'])
            y_prob.append(prediction.detach().numpy())
            y_true.append(batch_data['labels'].numpy().astype(int))

    y_true = y_true[0].reshape(-1)
    y_prob = y_prob[0]

    score = top_k_accuracy_score(y_true, y_prob, k=2)
    metrics_result['top_k_accuracy_score'] = score

    TOP_N = 10
    recommended_list_top10 = []

    content_embedding = trained_model.state_dict()['emb_layers.contentId.weight']
    user_embedding = trained_model.state_dict()['emb_layers.personId.weight']


    for person_id in test.index:
        user_emb = user_embedding[person_id]
        content_score = utils.compute_scores(user_emb, content_embedding)
        desc_indices = np.flip(np.argsort(content_score))[: TOP_N]
        recommended_list_top10.append(desc_indices)

    test['nn_predictions'] = recommended_list_top10

    mean_average_recall_K = []
    for K in np.arange(1, 11):
        mean_average_recall_K.extend([recmetrics.mark(test.actual.values.tolist(), test.nn_predictions.values.tolist(), k=K)])
    metrics_result['mean_average_recall_K'] = mean_average_recall_K

    personalizs_scroe = recmetrics.personalization(test.nn_predictions.values.tolist())

    metrics_result['personalizs_scroe'] = personalizs_scroe
    
    return metrics_result


