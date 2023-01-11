import sys
import pandas as pd
import pickle, json

from torch.utils.data import Dataset
from functools import reduce
from transformers import BertTokenizer

import numpy as np
import torch, sys, argparse, time
sys.path.append('./modeling')
import models as bm
import data_utils, model_utils, datasets
import input_models as im
import torch.optim as optim
import torch.nn as nn

SEED  = 1234
use_cuda = False

def read_config(path="../config/config-tganet.txt"):
    config = dict()
    with open(path, 'r') as f:
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    return config

# {'topic_rep_dict': '../resources/topicreps//bert_tfidfW_ward_euclidean_197-train.labels.pkl'
vocab_name = "../resources/glove.6B.100d.vocab.pkl"
keep_sen = False
is_bert = True
add_special_tokens = True
max_sen_len=10 
max_tok_len=200 
max_top_len=5
pad_value = 0
config = {'name': 'tganet_t7', 'ckp_path': '../checkpoints/', 'res_path': '../data/', 'b': '64', 
        'text_dim': '768', 'use_ori_topic': '0', 'epochs': '50', 'together_in': '1', 'att_mode': 'topic_only', 
        'topic_name': 'bert_tfidfW_ward_euclidean_197', 'max_tok_len': '200', 'topic_dim': '1536', 
        'bert': 'yes', 'max_sen_len': '10', 'max_top_len': '5', 'topic_path': '../resources/topicreps/', 
        'add_resid': '0', 'in_dropout': '0.35000706311476193', 'hidden_size': '401'}

trn_data_kwargs = {}
dev_data_kwargs = {}
topic_vecs = np.load('{}/{}.{}.npy'.format(config['topic_path'], config['topic_name'], config.get('rep_v', 'centroids')))
trn_data_kwargs['topic_rep_dict'] = '{}/{}-train.labels.pkl'.format(config['topic_path'], config['topic_name'])
dev_data_kwargs['topic_rep_dict'] = '{}/{}-dev.labels.pkl'.format(config['topic_path'], config['topic_name'])
word2i = {}

def print_cols(df):
    return
    l = list( df.columns )
    l.sort()
    print("print_cols -----------------")
    print(l)
    for i in l:
        if 'text' in i:
            print(f"{i}:")
            print( df[i][0] )
    print("---")

def get_index(word):
    if type(word) == list:
        l = []
        for w in word:
            l.append( get_index(w) )
        return l
    if word not in word2i:
        word2i[ word ] = len(word2i)
    return word2i[word] 

def flatten(l):
    return [item for sublist in l for item in sublist]


# Methods to use in apply
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, truncation=True) # added truncation = True to stop error
def mybuild(s):
    return tokenizer.build_inputs_with_special_tokens(s.text_idx, s.topic_idx)

def mycreate(s):
    return tokenizer.create_token_type_ids_from_sequences(s.text_idx, s.topic_idx)


def preprocess_data( data_name='../data/VAST/vast_train.csv',  ):
    #global word2i
    data_file = pd.read_csv(data_name)
    ##print_cols( data_file )
    #if vocab_name != None:
    #    word2i = pickle.load(open(vocab_name, 'rb'))

    print('preprocessing data {} ...'.format(data_name))
    if is_bert:
        print( data_file["topic_str"] )

        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, truncation=True) # added truncation = True to stop error
        print("processing BERT")
        data_file['text'] = data_file['text'].apply(json.loads) # asssumed to be deserialized list of lists
        # Shorten the text and topic data using limits from original code
        data_file['text'] = data_file['text'].apply(lambda text: text[:max_sen_len]) # JGM MAX
        data_file['text'] = data_file['text'].apply(lambda text: [t[:max_tok_len] for t in text]) # JGM MAX
        data_file['topic'] = data_file['topic'].apply(json.loads) # same assumption
        data_file['topic'] = data_file['topic'].apply(lambda x: x[:max_top_len]) # JGM MAX

        data_file['ori_topic'] = data_file['topic'] ##.apply(json.loads) # same assumption
        data_file['num_sens'] = data_file['text'].apply(len) # count of sentences
        data_file['ori_text'] = data_file['text'].apply( lambda x: ' '.join( flatten(x) ) ) # seems to be regular sentence (lowercase,cleaned)
        data_file['text_idx'] = data_file['ori_text'].apply(
            lambda x: tokenizer.encode(x, add_special_tokens=add_special_tokens, max_length=max_tok_len, pad_to_max_length=True)
        ) # regular sentence embedded
        data_file['topic_idx'] = data_file['ori_topic'].apply(
            lambda x: tokenizer.encode(x, add_special_tokens=add_special_tokens, max_length=max_tok_len, pad_to_max_length=True)
        ) # regular topic embedded
        print( 'text_idx --------' )
        print( type( data_file['text_idx'].values) )
        data_file['text_topic_batch'] = data_file.apply( lambda x: mybuild(x), axis=1 )
        data_file['token_type_ids'] = data_file.apply( lambda x: mycreate(x), axis=1 )
        print(f"...finished pre-processing for BERT {type(data_file)}")
        return data_file

from torch.utils.data import Dataset, DataLoader

class DataSlice(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        s = self.df.iloc[ index ]
        val = self.df['text_topic_batch'].values
        print(f"*** text_topic_batch: {val}")
        print(f"*** text_topic_batch: {type(val)}")
        val = np.array([np.array(xi,dtype=float) for xi in val])
        t = torch.from_numpy(val)
        print(f"*** text_topic_batch: {t}")
        print(f"*** text_topic_batch: {t.size()}")
        d = s.to_dict() #orient='records')
        nd = {}
        for k,v in d.items():
            if k not in ['text_topic_batch','token_type_ids','text_idx','topic_idx','topic_rep_ids']:
                continue
            print(f"DataSlice {k} >>> {v}")
            if k == 'text_topic_batch':
                nd[ k ] = torch.tensor( t )
            else:
                nd[ k ] = torch.tensor( v )
        return nd
    def keys(self):
        return self[0].keys()


def prepare_batch( df, **kwargs ):
    print( "prepare_batch - in work.py" )
    topic_batch = torch.Tensor( list( df['topic_idx'].values ) ) 
    text_batch = torch.Tensor( list( df['text_idx'].values ) )  
    df['labels'] = df['label'] # From CSV
    df['txt_l'] = df['text_idx'].apply(lambda x: sum( np.array( x ) != 0 ) ) 
    df['top_l'] = df['topic_idx'].apply(lambda x: sum( np.array( x ) != 0 ) ) 
    #df['ori_text'] = df['text_s'] # from CSV
    ds = DataSlice( df )
    return ds


def rename_variables(row): #from datasets
    """
    The grad students wrote hairball code. Now I need to rename everything into their
    clashing and inconsistent choices.
    """
    l= None
    sample = {'text': row['text_idx'], 'topic': row['topic_idx'], 'label': l,
              'txt_l': row['text_l'], 'top_l': row['topic_l'],
              'ori_topic': row['topic_str'],
              'ori_text': row['ori_text'],
              'num_s': row['num_sens'],
              'text_mask': row['text_mask'],
              'seen': row['seen?'],
              'id': row['new_id'],
              'contains_topic?': row['contains_topic?']} # HERE


data_file = preprocess_data()

# This will be in a loop
if True:
    print("Run the TGANet model!")
    batch_args = {'keep_sen': False}
    input_layer = im.JointBERTLayerWithExtra(vecs=topic_vecs, use_cuda=use_cuda,
                                                 use_both=(config.get('use_ori_topic', '1') == '1'),
                                                 static_vecs=(config.get('static_topics', '1') == '1'))

    setup_fn = data_utils.setup_helper_bert_attffnn

    loss_fn = nn.CrossEntropyLoss()

    model = bm.TGANet(in_dropout_prob=float(config['in_dropout']),
                             hidden_size=int(config['hidden_size']),
                             text_dim=int(config['text_dim']),
                             add_topic=(config.get('add_resid', '0') == '1'),
                             att_mode=config.get('att_mode', 'text_only'),
                             topic_dim=int(config['topic_dim']),
                             learned=(config.get('learned', '0') == '1'),
                             use_cuda=use_cuda)

    optimizer = optim.Adam(model.parameters())

    kwargs = {'model': model, #'embed_model': input_layer, 'dataloader': dataloader,
              #'batching_fn': data_utils.prepare_batch,
              'batching_kwargs': batch_args, 'name': config['name'],# + args['name'],
              'loss_function': loss_fn,
              'optimizer': optimizer,
              #'setup_fn': setup_fn
              }

    # -------
    # My modelling start
    print("Run the model!")
    filename='../checkpoints/ckp-tganet_t7-BEST.tar'
    checkpoint = torch.load(filename)
    #print( checkpoint )
    model.load_state_dict( checkpoint['model_state_dict'] )
    # TODO - load data, process and run in model
    # if data doesn't work use dataloader
    ##input_data, true_labels, id_lst = foo( dataloader ) #prepare_data(data, type_lst=type_lst)


    data_file['txt_l'] = sum(data_file['text_idx']!=0)  # Count the number of non-zero word indexes
    data_file['top_l'] = sum(data_file['topic_idx']!=0)  # Count the number of non-zero word indexes
    ##samples = data_file.to_dict(orient='records')

    #args = prepare_batch( sample_batched, **batch_args )
    ds = prepare_batch( data_file, **batch_args )
    args = ds[0]
    print( "ARGS:" )
    print( ds.keys() )
    print( type(args) )

    # EMBEDDING
    print(f"embed_args: {args}")
    embed_args = input_layer(**args) # embed model in model_utils
    print( embed_args )
    args.update(embed_args)
    print("DONE to input_layer")
    sys.exit()

    # PREDICTION
    print("START PREDICTION!")
    print( "ARGS:" )
    device = 'cuda' if use_cuda else 'cpu'
    txt_E  = args['txt_E'].to(device)  # (B,E)
    top_E  = args['top_E'].to(device)  # (B,E)
    top_rep = args['avg_top_E'].to(device)
    text_l = args['txt_l'].to(device)

    # PREDICTION
    print("START PREDICTION!")
    print( "ARGS:" )
    model.eval()
    ##y_pred = model( suf[0], suf[1], suf[2], suf[3] )
    y_pred = model( txt_E, top_E, top_rep, text_l )
    print( "y_pred" )
    pred_labels = y_pred.argmax(axis=1)
    print( pred_labels )
    ##print( y_pred )
    sys.exit()

    if isinstance(y_pred, dict):
        print("ONE")
        y_pred_arr = y_pred['preds'].detach().cpu().numpy()
    else:
        print("TWO")
        y_pred_arr = y_pred.detach().cpu().numpy()
    ##ls = np.array(labels)

    ##args['text_l'] = args['txt_l']
    """
    kp = ['text', 'topic', 'topic_rep', 'text_l']
    d = []
    for k in args.keys():
        if k not in kp:
            d.append(k)
    for i in d:
        del args[i]
    args['topic_rep'] = 'foo'
    print( args.keys() )

    ##pred_labels = model.predict(id_lst)
    #y_pred = model(*setup_fn(args, use_cuda))
    y_pred = model(**args)
    #y_pred = model(sample_batched)
    print( "y_pred" )
    print( y_pred )
    """
    sys.exit()
    # END
    # -------
    
