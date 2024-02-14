import torch
from src.config import Config

torch.set_num_threads(Config['training_setting']['num_threads'])


class LNBDR(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, use_bn=True):
        super(LNBDR, self).__init__()
        self.use_bn = use_bn
        self.linear = torch.nn.Linear(in_features, out_features)
        if use_bn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        if self.use_bn:
            return self.relu(self.dropout(self.bn(self.linear(inputs))))
        else:
            return self.relu(self.dropout(self.linear(inputs)))

# TODO could try output with dot product and compute loss (mse) 
# between the product and rating score (floating number as opposed to classification) e.g. movie sore 2, 4, 5.5
# https://www.kaggle.com/code/mejbahahammad/movie-recommender-systems-using-neural-network/input?select=ratings.csv
class DNN(torch.nn.Module):
    def __init__(self, mapping_dict, continuous_variables, catagory_variables, \
                    num_output, num_mlp_layer= 3, embedding_dim= 10, use_bn= True, ):
        super(DNN, self).__init__()
        self.continuous_variables = continuous_variables
        self.catagory_variables = catagory_variables
        self.embedding_dim = embedding_dim
        self.use_bn = use_bn
        self.mlp_size = self.embedding_dim * len(self.catagory_variables) + len(self.continuous_variables)    

        self.linear_size = [int(self.mlp_size / 2**i) for i in range(num_mlp_layer)]
        self.emb_layers = torch.nn.ModuleDict({x: torch.nn.Embedding(len(mapping_dict[x]), embedding_dim, padding_idx=0) for x in catagory_variables})

        self.mlp_layers = torch.nn.ModuleList([
            LNBDR(self.linear_size[i], self.linear_size[i+1], dropout=0.5, use_bn= self.use_bn)
                for i in range(len(self.linear_size)-1)
            ])

        self.mlp_layers.append(torch.nn.Linear(self.linear_size[-1], num_output))
        self.softmax = torch.nn.Softmax()

    

    def forward(self, inputs):
        emb_tensor_list = [self.emb_layers[x](inputs[x].type(torch.LongTensor)) for x in self.catagory_variables]
        mlp_input_tensors = torch.concat(emb_tensor_list  + [inputs[x].unsqueeze(1) for x in self.continuous_variables], dim=1)

        for l in self.mlp_layers:
            mlp_input_tensors = l(mlp_input_tensors)        
        vector = self.softmax(mlp_input_tensors)

        return vector


if __name__ == "__main__":
    pass
