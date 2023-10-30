#!/usr/bin/env python
# coding: utf-8

## This version differs from the previous in trying to reduce the execution time.
## First change is to alter the non-leaf factor ascription to be in batch.
"""
Change log:
v1_2 = Allows code to be run for various levels of abstraction of data input (i.e.,
    [leaf, intermediate, issue, outcome].
v1_1 = Added F1 score metric. Added functionality to record results, picking out best
    metric performance to write. Stores model of best F1 score for each node.
v1_0 = Changes focus to check for ascription performance rather than for classifying
    outcome. Overhauls the hybrid_angelic code to run over nodes first and epochs second
    which means that models no longer are stored. This facilitates parellel jobs. This
    code is obviously designed to run with angelic_design v2_n (where n is an integer)
    which has def backward_pass provide classification weights as a tuple of three weights,
    where the first is positive ascription, the second negative ascription, and the third
    indicates no ascription.
"""

import copy, itertools, math

import random as r
r.seed(2019)

import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

import os
os.environ['PYTHONHASHSEED'] = str(2019)

from util import d, here

import pandas as pd
pd.options.display.max_columns = None
from argparse import ArgumentParser
from datetime import datetime

import random, sys, math, gzip
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

import numpy as np
#np.random.seed(2019)
np.set_printoptions(threshold=sys.maxsize)
from numpy import genfromtxt

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

import gc
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.utils import shuffle
import glob
from optparse import OptionParser

import pickle
import shutil

## Importing ADF python files (should be kept in the base directory).
import s8_angelic_design_v2_2 as angel

args = {
    ## directory for encoded embeddings
    'data_dir': 'datasets/roberta-base_data/',
    ## output of the experiments
    'output_dir': 'outputs/',
    ## output of attention scores for sentences
    'attention_output_dir': 'attentions/',
    'cuda_num': 1,
}


 ## take a list of strings as optional arguments
def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class Normalize(object):
    
    def normalize_train(self, X_train, max_len):
        self.scaler = MinMaxScaler()
        X_train = X_train.reshape(X_train.shape[0],-1)
        X_train = self.scaler.fit_transform(X_train)
        X_train = X_train.reshape(X_train.shape[0],max_len,-1)
        return X_train
    
    def normalize_test(self, X_test, max_len):
        X_test = X_test.reshape(X_test.shape[0],-1)
        X_test = self.scaler.transform(X_test)
        X_test = X_test.reshape(X_test.shape[0],max_len,-1)
        return X_test

    def inverse(self, X_train, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_test   = self.scaler.inverse_transform(X_test)    
        return (X_train, X_test)


# ## Config model

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


class BertConfig():

    """
        :class:`~transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.
        Arguments:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            seq_length: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=4,
                 num_attention_heads=1,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.01,
                 attention_probs_dropout_prob=0.01,
                 seq_length=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 output_attentions=True,
                 output_hidden_states=False,
                 num_labels=2):

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.seq_length = seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_labels = num_labels


# ## Customized Sentence-level Transformer
BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """input sentence embeddings inferred by bottom pre-trained BERT, contruct location embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.location_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds, location_ids=None):
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = inputs_embeds.device
        if location_ids is None:
            location_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            location_ids = location_ids.unsqueeze(0).expand(input_shape)
        location_embeddings = self.location_embeddings(location_ids)
        embeddings = inputs_embeds + location_embeddings 
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self,config):
        super(BertAttention, self).__init__()
        self.config = config
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        
    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = torch.nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)                   
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):

        # We "pool" the model by simply averaging hidden states
        mean_tensor = hidden_states.mean(dim=1)
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple       
    """  
    
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
 
    def forward(self, inputs_embeds, attention_mask=None, location_ids=None):
        """ Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
            
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
        input_shape = inputs_embeds.size()[:-1]
        device = inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
            
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))
            
        # Since attention_mask is 1.0 for locations we want to attend and 0.0 for
        # masked locations, this operation will create a tensor which is 0.0 for
        # locations we want to attend and -10000.0 for masked locations.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(inputs_embeds=inputs_embeds, location_ids=location_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class HTransformer(nn.Module):
    """
    Sentence-level transformer, several transformer blocks + softmax layer   
    """
    def __init__(self, config):
        """
        :param emb_size: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_classes: Number of classes.   
        """
        super(HTransformer,self).__init__()       
        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            location_ids=None,
                            inputs_embeds=x)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]        
        return outputs


def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


if __name__ == "__main__":
    
    parser = OptionParser(usage='usage: -a abstract_level -r random_seeds -d dataset_name -l learning_rate -e no_epochs -m max_len -p test_prop')   
    parser.add_option("-d", "--dataset_name", action = "store", type = "string", dest = "dataset_name", help = "directory of data encoded by token-level Roberta", default = 'article6')
    parser.add_option("-a", "--abstract_level", action = "store", type = "string", dest = "abstract_level", help = "abstraction level in [Leaf, Intermediate, Issue, Outcome]", default = 'Intermediate')
    parser.add_option("-l", "--learning_rate", action = "store", type = "float", dest = "learning_rate", help = "learning rate for fine tuning", default = 2e-6)
    parser.add_option("-e", "--no_epochs", action = "store", type = "int", dest = "no_epochs", help = "the number of epochs for fine tuning", default = 30)
    parser.add_option("-m", "--max_len", action = "store", type = "int", dest = "max_len", help = "the maximum number of bullets per document", default = 256)
    parser.add_option('-r', "--random_seed", type = 'int', action = 'store', dest = 'random_seed', help = "the random seed", default = 1988)
    parser.add_option('-p', "--test_prop", type = 'float', action = 'store', dest = 'test_prop', help = "the proportion of the data used for training", default = 0.2)
    (options, _) = parser.parse_args()
 
    dataset = options.dataset_name
    abstraction = options.abstract_level
    lr = options.learning_rate
    no_epochs = options.no_epochs
    max_len = options.max_len
    seed = options.random_seed
    test_prop = options.test_prop
    
    random.seed(seed)

    script_name = os.path.basename(__file__)[:-3]
    exp_name = script_name + '_' + abstraction
    save_name = '%s_epoch%s_prop%s_len%s/'%(script_name, no_epochs, test_prop, max_len)
    unique_id = datetime.today().strftime('%Y-%m-%d-%Hhr%M')
    
    print("unique_id: ", unique_id)   
    print('dataset name: ', dataset)
    print('number of epochs: ', no_epochs)
    print('initial random state: ', seed)
    print("proportion of data for test: ", test_prop)
    print("max no. of bullets: ", max_len, '\n')

    gradient_clipping = 1.0
    train_batch = 16 
    eval_batch = 32
    
    ## Use angelic_design.py to read the angelic design contained in the
    ## art6_angelic_design.csv file, in order to ascribe factors -> outcome
    ## in the forward pass, and return classification weights in the backward pass.
    assert dataset == "article6"
    A_df = angel.create_AD("art6_angelic_design.csv")
    assert (A_df['Acceptance'].isna()).all()
    
    ## Importing annotated data sets, detailing the students' case ascriptions 
    ## produced via the Turing funded research project.
    non_label_df = angel.imported_acceptance("non")
    vio_label_df = angel.imported_acceptance("vio")

    ## Importing the RoBERTa encodings for the natural language descriptions of the cases.
    with open(os.path.join(args['data_dir'], dataset + '_non.p'), 'rb') as fp:
        pre_trained_non_dict = pickle.load(fp)
    with open(os.path.join(args['data_dir'], dataset + '_vio.p'), 'rb') as fp:
        pre_trained_vio_dict = pickle.load(fp)

    ## Setting data set to January 2015 data. 
    ## January 2015 non id: '001-150317'; vio id: '001-149205'
    ## January 2021 non id: '001-207129'; vio id: '001-207408'
    ## Debugging non id: '001-210281'; vio id: '001-213200'
    def get_relevant_data(target_id, raw_trained_dict):
        relevant_dict = dict()
        for case_id in raw_trained_dict.keys():
            assert case_id[:3] == '001'
            float_identifier = float(case_id[-6:])
            if float_identifier >= float(target_id[-6:]):
                relevant_dict[case_id] = raw_trained_dict[case_id]
        return relevant_dict                     
    pre_trained_non_dict = get_relevant_data('001-150317', pre_trained_non_dict)
    pre_trained_vio_dict = get_relevant_data('001-149205', pre_trained_vio_dict)
    
    ## Using def backward_pass in angelic_design.py to create a dictionary 
    ## of node weights for each case. And checking that each case has 
    ## either a corresponding annotation in non_label_df or vio_label_df.
    def get_case_node_weights(case_encodings, annotations, outcome):
        angel_weights = dict() 
        case_id_list = []
        missing_ids = []
        #print(annotations.keys())
        for case_id in case_encodings.keys():
            if case_id not in annotations.keys():
                missing_ids.append(case_id)
            case_id_list.append(case_id)
            instance_df = annotations[case_id]
            if outcome == 'non':
                instance_df['VIOLATION'] = [0, 1, 0]
            elif outcome == 'vio':
                instance_df['VIOLATION'] = [1, 0, 0]
            else:
                print('Outcome needs to be non or vio')
                sys.exit()
            node_weights = angel.backward_pass(A_df, instance_df, case_id, abstraction)
            angel_weights.update({case_id: node_weights})
        print("No. of missing case encoding case ids from annotation set:", len(missing_ids), '\n')
        return angel_weights, case_id_list

    angel_weights_non, case_ids_non = get_case_node_weights(pre_trained_non_dict, non_label_df, 'non')  
    angel_weights_vio, case_ids_vio = get_case_node_weights(pre_trained_vio_dict, vio_label_df, 'vio')
    #print("angel_weights_non:", angel_weights_non, '\n')
   
    ## Get list of all case_ids
    all_case_ids = case_ids_non + case_ids_vio
    
    ## Setting the output_df that will store the ascriptions across all legal nodes across
    ## all cases (training, validation, and test sets).
    output_df = pd.DataFrame(columns = ['Factor', 'Type'] + all_case_ids)
    #print(A_df[['Factor', 'Type']])
    output_df[['Factor', 'Type']] = A_df[['Factor', 'Type']]
     
    ## Setting up non-violation and violation data sets.
    def process_raw_data(encoding_dict, outcome): 
        reshaped_dict = dict()    
        for case_id in encoding_dict.keys():  
            case_array = np.zeros((max_len, 768))
            doc = encoding_dict[case_id]
            if doc.shape[0] <= max_len:
                case_array[:doc.shape[0], :] = doc
            else:
                case_array[:max_len, :] = doc[:max_len, :]
            reshaped_dict[case_id] = case_array          
        return reshaped_dict

    data_non = process_raw_data(pre_trained_non_dict, 'non-violation')
    data_vio = process_raw_data(pre_trained_vio_dict, 'violation')
    data_non_len = len(data_non)
    data_vio_len = len(data_vio)
    print('data_non_len:', data_non_len)
    print('data_vio_len:', data_vio_len)
    
    ## Function to split dictionaries into different sets (e.g., split into training and test sets).
    def split_data(input_dict, split_prop):
        keys = list(input_dict.keys())
        keys_train, keys_test = train_test_split(keys, test_size = split_prop)
        train_dict = {key: input_dict[key] for key in keys_train}
        test_dict = {key: input_dict[key] for key in keys_test}
        return train_dict, test_dict
    
    ## Splitting initial data into test set and learning set - with learning set
    ## to be later further split into training and validation sets.
    train_non, test_non = split_data(data_non, test_prop)
    train_vio, test_vio = split_data(data_vio, test_prop)
    train_non_len = len(train_non)
    train_vio_len = len(train_vio)
    test_non_len = len(test_non)
    test_vio_len = len(test_vio)
    print("train_non_len: ", len(train_non))
    print("train_vio_len: ", len(train_vio))
    print("test_non_len: ", test_non_len)
    print("test_vio_len: ", test_vio_len, '\n')
    assert data_non_len == train_non_len + test_non_len
    assert data_vio_len == train_vio_len + test_vio_len
    
    ## Identifying the nodes in the angelic design.
    config = BertConfig(seq_length = max_len)
    nodes = A_df.loc[A_df['Type'] == abstraction, 'Factor'].tolist() ## nodes is the list of legal nodes used throughout

    print("Setup completed. Ready to start!", '\n')
       
    """ 
    def create_dataset determines case ascription for a node and produces training or test data set.
    """
    def create_dataset(non_case_data, vio_case_data, eval_type):      
        """      
        doc_ascribe is used to determine whether a factor is ascribed to a particular case
        based on its angelic weights.
        """
        def doc_ascribe(outcome):
            if outcome == 'non':
                node_angel_weights = angel_weights_non[case_id][node]
            elif outcome == 'vio':
                node_angel_weights = angel_weights_vio[case_id][node]
            else:
                print('Outcome must be in [non, vio]')
                sys.exit()
            pos_weight = node_angel_weights[0] + node_angel_weights[1]
            neg_weight = node_angel_weights[2]
            if random.uniform(0, 1) <= pos_weight:
                pos_ascribe.append(case_encoding)
                ascription_dict[case_id] = 1
            if random.uniform(0, 1) <= neg_weight:
                neg_ascribe.append(case_encoding)  
                ascription_dict[case_id] = 0
            return pos_ascribe, neg_ascribe, ascription_dict
            
        ## Training the model for each node, i.e., each base-level factor.
        neg_ascribe = []
        pos_ascribe = []
        ascription_dict = dict()
        for case_id, case_encoding in non_case_data.items():
            pos_ascribe, neg_ascribe, ascription_dict = doc_ascribe("non")
        for case_id, case_encoding in vio_case_data.items():
            pos_ascribe, neg_ascribe, ascription_dict = doc_ascribe("vio")
        len_neg_ascribe = len(neg_ascribe)
        len_pos_ascribe = len(pos_ascribe)
        
        ## Rendering data set as array and concatenating as appropriate (including oversampling
        ## for training data).
        if len_neg_ascribe == 0 and len_pos_ascribe == 0:
            X_set = np.array([])
            y_set = np.array([])
            print('Data set should not be empty!')
            sys.exit()
        elif len_neg_ascribe == 0:
            X_set = np.asarray(pos_ascribe)
            y_set = np.array([1] * len_pos_ascribe)
        elif len_pos_ascribe == 0:
            X_set = np.asarray(neg_ascribe)
            y_set = np.array([0] * len_neg_ascribe)            
        else:
            if eval_type == 'train':
                len_diff = abs(len_pos_ascribe - len_neg_ascribe)
                if len_pos_ascribe < len_neg_ascribe:
                    pos_ascribe += pos_ascribe * (len_diff // len(pos_ascribe)) + pos_ascribe[:len_diff % len(pos_ascribe)]
                elif len_neg_ascribe < len_pos_ascribe:
                    neg_ascribe += neg_ascribe * (len_diff // len(neg_ascribe)) + neg_ascribe[:len_diff % len(neg_ascribe)]                
                assert len(neg_ascribe) == len(pos_ascribe)
            else:
                assert eval_type == 'test'
            X_set = np.concatenate((np.asarray(neg_ascribe), np.asarray(pos_ascribe)))
            y_set = np.array([0] * len(neg_ascribe) + [1] * len(pos_ascribe))
        num_instances = len(X_set)
        assert num_instances == len(y_set)
        if len(X_set) > 0:
            assert X_set.shape == (num_instances, max_len, 768)
        
        ## Normalising and creating batches.
        if eval_type == 'train':
            X_set = normalizer.normalize_train(X_set, max_len)
        elif eval_type == 'test':
            X_set = normalizer.normalize_test(X_set, max_len)
        else:
            print('Data must be training or test')
            sys.exit()
        tensor_set_x = torch.from_numpy(X_set).type(torch.FloatTensor)
        tensor_set_y = torch.from_numpy(y_set).type(torch.LongTensor)
        del X_set, y_set
        eval_data_set = torch.utils.data.TensorDataset(tensor_set_x, tensor_set_y)
        if eval_type == 'train':
            dataset_loader = torch.utils.data.DataLoader(eval_data_set, batch_size = train_batch, shuffle = True, num_workers = 1)
        elif eval_type == 'test':
            dataset_loader = torch.utils.data.DataLoader(eval_data_set, batch_size = eval_batch, shuffle = False, num_workers = 1)
        else:
            print('Data must be training or test')
            sys.exit()
        return dataset_loader, ascription_dict
    
    """
    def node_ascribe is used to ascribe a particular node across all cases in an input data set.
    """
    def node_ascribe(set_loader):
        with torch.no_grad():
            model.train(False)
            fact_atten_scores = torch.Tensor()
            y_pred = []
            y_true = []
            for idx,data in enumerate(set_loader):
                inputs, labels = data
                if inputs.size(1) > config.seq_length:
                    inputs = inputs[:, :config.seq_length, :]
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda(args['cuda_num'])), labels.cuda(args['cuda_num'])
                out = model(inputs)
                sm = nn.Softmax(dim = 1)
                pred_prob = out[0].cpu()
                pred_prob = sm(pred_prob)
                predict = torch.argmax(pred_prob, axis = 1)
                labels = labels.cpu()
                y_pred = y_pred + predict.tolist()
                y_true = y_true + labels.tolist()
                del inputs, labels, out
        return y_pred, y_true
    
    """
    def eval_stats produces two similarly ordered lists across a type of input data set (i.e.,
    training, validation, or test). One list is the target classifications, and the other
    list is the output predictions.
    """
    def eval_stats(y_eval_pred, y_eval_true, eval_type): 
        eval_acc = accuracy_score(y_eval_true, y_eval_pred)
        eval_f_score = f1_score(y_eval_true, y_eval_pred, average = 'binary')
        eval_tn, eval_fp, eval_fn, eval_tp = confusion_matrix(y_eval_true, y_eval_pred, labels = [0, 1]).ravel()
        eval_mcc = matthews_corrcoef(y_eval_true, y_eval_pred)
        print('%s accuracy:'%(eval_type), eval_acc)
        print('%s F1 Score:'%(eval_type), eval_f_score)
        print('%s MCC:'%(eval_type), eval_mcc)
        print('%s confusion matrix:'%(eval_type), 'TP', eval_tp, 'TN', eval_tn, 'FP', eval_fp, 'FN', eval_fn)
        return eval_f_score, eval_mcc, eval_tp, eval_tn, eval_fp, eval_fn
    
    """    
    Running the experimentations over the legal nodes and the epochs of learning.
    """
    results_dirs = os.path.join(args['output_dir'],'%s'%(dataset), exp_name, unique_id) 
    if not os.path.exists(results_dirs):
        os.makedirs(results_dirs)
    for node in nodes:
        print('node:', node, '\n')
    
        ## Initialising the analytics
        train_factor_ascribe = dict()
        train_confusion_matrices = []
        train_f1_scores = []
        train_mcc_scores = []
        test_factor_ascribe = dict()
        test_confusion_matrices = []
        test_f1_scores = []
        test_mcc_scores = []
        losses = []
        macro_f = []
        train_best_f1 = -1.0
        test_best_f1 = -1.0
        
        ## Initialising the node model, normalizer, and the data sets.
        model = HTransformer(config = config)
        model.apply(init_weights)
        model.cuda(1)
        opt = torch.optim.Adam(lr = lr, params = model.parameters())
        normalizer = Normalize()
        trainloader, train_ascription = create_dataset(train_non, train_vio, 'train')
        testloader, test_ascription = create_dataset(test_non, test_vio, 'test')
        
        ## Run experiment on legal node over epochs.
        for e in tqdm(range(no_epochs)):
            print('\n epoch ',e)
            
            ## Train model on batches.
            for i, data in enumerate(trainloader):
                model.train(True)
                opt.zero_grad()
                inputs, labels = data
                if inputs.size(1) > config.seq_length:
                    inputs = inputs[:, :config.seq_length, :]
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda(1)), labels.cuda(1)
                out = model(inputs)
                weight = [1.0, 1.0]
                weight = torch.tensor(weight).cuda(1)
                loss = nn.CrossEntropyLoss(weight,reduction = 'mean')
                output = loss(out[0], labels)
                train_loss_tol = float(output.cpu())
                output.backward()
                if gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                opt.step()
                del inputs, labels, out, output
                torch.cuda.empty_cache()
                losses.append(train_loss_tol)   
                    
            ## Analysis of training set. 
            y_train_pred, y_train_true = node_ascribe(trainloader)
            train_f_score, train_mcc, train_tp, train_tn, train_fp, train_fn = eval_stats(y_train_pred, y_train_true, 'training')
            train_mcc_scores.append(train_mcc)
            train_confusion_matrices.append({'TP':train_tp, 'TN':train_tn, 'FP':train_fp, 'FN':train_fn})
            print('\n')
            
            ## Analysis of test set.  
            y_test_pred, y_test_true = node_ascribe(testloader)
            test_f_score, test_mcc, test_tp, test_tn, test_fp, test_fn = eval_stats(y_test_pred, y_test_true, 'test')
            test_mcc_scores.append(test_mcc)
            test_confusion_matrices.append({'TP':test_tp, 'TN':test_tn, 'FP':test_fp, 'FN':test_fn})
            print('\n')

            ## Recording best models. Note if the current best pos f1 score is smaller than the
            ## current epoch MCC score, we will update the attention scores.
            best_models_dirs = os.path.join('best_models','%s'%(dataset), exp_name)
            if not os.path.exists(best_models_dirs):
                os.makedirs(best_models_dirs)
                
            ## Storing best models.
            if test_best_f1 < test_f_score:
                suffix = '.pickle'
                old_f1 = -1.0
                #print(test_f_score, 'is better than', test_best_f1)
                test_best_f1 = test_f_score  
                for filename in os.listdir(best_models_dirs):
                    if filename.startswith(node + "_f1_"):
                        #print('found', filename)
                        old_f1 = float(filename.split("_")[-1][:-len(suffix)])
                        if test_f_score > old_f1:
                            os.remove(os.path.join(best_models_dirs, filename))
                if test_f_score > old_f1:
                    #print('Model to be saved.')
                    new_filename = "%s_f1_%s%s" % (node, test_f_score, suffix)
                    write_path = os.path.join(best_models_dirs, new_filename)
                    with open(write_path, 'wb') as fp:
                        pickle.dump(model, fp, pickle.HIGHEST_PROTOCOL)
                #print('\n')

        """
        Recording results for node.
        """
        print('Best F1 Score:', test_best_f1)
        ## Recording the results.
        with open(os.path.join(results_dirs, "%s_test_confusion.pickle"%(node)), "wb") as fp:
            pickle.dump(test_confusion_matrices, fp)  




