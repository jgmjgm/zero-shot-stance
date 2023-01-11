import numpy as np
import torch, sys, argparse, time
sys.path.append('./modeling')
import models as bm
import data_utils, model_utils, datasets
import input_models as im
import torch.optim as optim
import torch.nn as nn

SEED  = 1234
NUM_GPUS = None
use_cuda = False #torch.cuda.is_available()
print( f"GPU available {use_cuda}" )

def print_args( args ):
    print(f"PRINTING ARGS:")
    keys = args.keys()
    for k in keys:
        val = args[k][1]
        s = f"{k}: {val}"
        if len(s) > 300:
            s = s[:300]
        print( s )

def foo( dataloader ):
    data = dataloader
    data = data.get()

    concat_data = []
    labels = []
    id_lst = []
    for s in data:
        #if type_lst is not None and s.get('seen', -1) not in type_lst: continue
        concat_data.append(s['text'] + s['topic'])
        labels.append(np.argmax(s['label']))
        id_lst.append(s['id'])

    input_data = np.array(concat_data)
    input_labels = np.array(labels)

    return input_data, input_labels, id_lst



def train(model_handler, num_epochs, verbose=True, dev_data=None,
          early_stopping=False, num_warm=0, is_bert=False):
    '''
    Trains the given model using the given data for the specified
    number of epochs. Prints training loss and evaluation starting
    after 10 epochs. Saves at most 10 checkpoints plus a final one.
    :param model_handler: a holder with a model and data to be trained.
                            Assuming the model is a pytorch model.
    :param num_epochs: the number of epochs to train the model for.
    :param verbose: whether or not to print train results while training.
                    Default (True): do print intermediate results.
    :param corpus_samplers: list of samplers for individual corpora, None
                            if only evaling on the full corpus.
    '''
    prev_dev_loss = 0
    for epoch in range(num_epochs):
        model_handler.train_step()

        if epoch >= num_warm:
            if verbose:
            # print training loss and training (& dev) scores, ignores the first few epochs
                print("training loss: {}".format(model_handler.loss))
                # eval model on training data
                if not is_bert:
                    # don't do train eval if bert, because super slow
                    trn_scores = model_handler.eval_and_print(data_name='TRAIN')
                # update best scores
                if dev_data is not None:
                    dev_scores = model_handler.eval_and_print(data=dev_data, data_name='DEV')
                    model_handler.save_best(scores=dev_scores)
                else:
                    model_handler.save_best(scores=trn_scores)
            if early_stopping:
                l = model_handler.loss # will be dev loss because evaled last
                if l < prev_dev_loss:
                    break

    print("TRAINED for {} epochs".format(epoch))

    if early_stopping:
        save_num = "BEST"
    else:
        save_num = "FINAL"

    # save final checkpoint
    model_handler.save(num=save_num)

    # print final training (& dev) scores
    model_handler.eval_and_print(data_name='TRAIN')
    if dev_data is not None:
        model_handler.eval_and_print(data=dev_data, data_name='DEV')



if __name__ == '__main__':
    SKIP = False # Skip the loading of data bc it takes a while

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-n', '--name', help='something to add to the saved model name',
                        required=False, default='')
    parser.add_argument('-e', '--early_stop', help='Whether to do early stopping or not',
                        required=False, type=bool, default=False)
    parser.add_argument('-p', '--num_warm', help='Number of warm-up epochs', required=False,
                        type=int, default=0)
    parser.add_argument('-k', '--score_key', help='Score to use for optimization', required=False,
                        default='f_macro')
    parser.add_argument('-v', '--save_ckp', help='Whether to save checkpoints', required=False,
                        default=0, type=int)
    args = vars(parser.parse_args())

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    ####################
    # load config file #
    ####################
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    ################
    # load vectors #
    ################
    vec_name = config.get('vec_name', '')
    vec_dim  = int(config.get('vec_dim', 0))

    if 'vec_name' in config:
        vecs = data_utils.load_vectors('../resources/{}.vectors.npy'.format(vec_name),
                                   dim=vec_dim, seed=SEED)

    trn_data_kwargs = {}
    dev_data_kwargs = {}

    if 'topic_name' in config:
        print("CONFIG !!!!!!!!!!!!!!!!!!!!!!!!")
        print( config )
        print("END !!!!!!!!!!!!!!!!!!!!!!!!")
        topic_vecs = np.load('{}/{}.{}.npy'.format(config['topic_path'], config['topic_name'], config.get('rep_v', 'centroids')))
        trn_data_kwargs['topic_rep_dict'] = '{}/{}-train.labels.pkl'.format(config['topic_path'], config['topic_name'])
        dev_data_kwargs['topic_rep_dict'] = '{}/{}-dev.labels.pkl'.format(config['topic_path'], config['topic_name'])

    #############
    # LOAD DATA #
    #############
    # load training data
    if 'bert' not in config and 'bert' not in config['name']:
        vocab_name = '../resources/{}.vocab.pkl'.format(vec_name)
        print( "VOCAB_NAME" )
        print( vocab_name )
        data = datasets.StanceData(args['trn_data'], vocab_name,
                                   pad_val=len(vecs) - 1,
                                   max_tok_len=int(config.get('max_tok_len', '200')),
                                   max_sen_len=int(config.get('max_sen_len', '10')),
                                   keep_sen=('keep_sen' in config),
                                   **trn_data_kwargs)
    else:
        print( "IN HERE" )
        data = None
        if not SKIP:
            data = datasets.StanceData(args['trn_data'], None, max_tok_len=config['max_tok_len'],
                                   max_top_len=config['max_top_len'], is_bert=True,
                                   add_special_tokens=(config.get('together_in', '0') == '0'),
                                   ##keep_sen=('keep_sen' in config), #NEW jgm
                                   **trn_data_kwargs)

    dataloader = None
    if not SKIP:
        dataloader = data_utils.DataSampler(data,  batch_size=int(config['b']))

    # load dev data if specified
    if args['dev_data'] is not None:
        if 'bert' not in config and 'bert' not in config['name']:
            print("THIS")
            dev_data = datasets.StanceData(args['dev_data'], vocab_name,
                                               pad_val=len(vecs) - 1,
                                           max_tok_len=int(config.get('max_tok_len', '200')),
                                           max_sen_len=int(config.get('max_sen_len', '10')),
                                           keep_sen=('keep_sen' in config),
                                           **dev_data_kwargs)
        else:
            print("THAT")
            dev_data = None
            if not SKIP:
                dev_data = datasets.StanceData(args['dev_data'], None, max_tok_len=config['max_tok_len'],
                                           max_top_len=config['max_top_len'], is_bert=True,
                                           add_special_tokens=(config.get('together_in', '0') == '0'),
                                           **dev_data_kwargs)

        dev_dataloader = None
        if not SKIP:
            dev_dataloader = data_utils.DataSampler(dev_data, batch_size=int(config['b']), shuffle=False)
    else:
        dev_dataloader = None


    ### set the optimizer
    lr = float(config.get('lr', '0.001'))

    # RUN
    print("Using cuda?: {}".format(use_cuda))

    if 'BiCond' in config['name']:
        batch_args = {}
        input_layer = im.BasicWordEmbedLayer(vecs=vecs, use_cuda=use_cuda,
                                                         static_embeds=(config.get('tune_embeds', '0') == '0'))

        setup_fn = data_utils.setup_helper_bicond
        loss_fn = nn.CrossEntropyLoss()

        model = bm.BiCondLSTMModel(hidden_dim=int(config['h']), embed_dim=input_layer.dim,
                                   input_dim=(int(config['in_dim']) if 'in_dim' in config['name'] else input_layer.dim),
                                   drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                   num_labels=3, keep_sentences=('keep_sen' in config),
                                   doc_method=config.get('doc_m', 'maxpool'))
        o = optim.Adam(model.parameters(), lr=lr)

        bf = data_utils.prepare_batch

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': bf,
                  'batching_kwargs': batch_args, 'name': config['name'] + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': o,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, num_gpus=NUM_GPUS,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path','data/gen-stance/'),
                                                      use_score=args['score_key'],save_ckp=(args['save_ckp'] == 1),
                                                      **kwargs)

    elif 'CTSAN' in config['name']:
        batch_args = {}
        input_layer = im.BasicWordEmbedLayer(vecs=vecs, use_cuda=use_cuda)
        setup_fn = data_utils.setup_helper_bicond

        loss_fn = nn.CrossEntropyLoss()

        bf = data_utils.prepare_batch

        model = bm.CTSAN(hidden_dim=int(config['h']), embed_dim=input_layer.dim, att_dim=int(config['a']),
                         lin_size=int(config['lh']), drop_prob=float(config['dropout']),
                         use_cuda=use_cuda, out_dim=3, keep_sentences=('keep_sen' in config),
                         sentence_version=config.get('sen_v', 'default'),
                         doc_method=config.get('doc_m', 'maxpool'),
                         premade_topic=('topic_name' in config),
                         topic_trans=('topic_name' in config),
                         topic_dim=(int(config.get('topic_dim')) if 'topic_dim' in config else None))

        o = optim.Adam(model.parameters(), lr=lr)

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': bf,
                  'batching_kwargs': batch_args, 'name': config['name'] + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': o,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, num_gpus=NUM_GPUS,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'],save_ckp=(args['save_ckp'] == 1),
                                                      **kwargs)

    elif 'ffnn-bert' in config['name']: # Joint BERT (similar to author's model)
        if config.get('together_in', '0') == '1':
            batch_args = {'keep_sen': False}
            if 'topic_name' in config:
                input_layer = im.JointBERTLayerWithExtra(vecs=topic_vecs, use_cuda=use_cuda,
                                                         use_both=(config.get('use_ori_topic', '1') == '1'),
                                                         static_vecs=(config.get('static_topics', '1') == '1'))
            else:
                input_layer = im.JointBERTLayer(use_cuda=use_cuda)

        else:
            batch_args = {'keep_sen': False}
            input_layer = im.BERTLayer(mode='text-level', use_cuda=use_cuda)

        setup_fn =data_utils.setup_helper_bert_ffnn

        loss_fn = nn.CrossEntropyLoss()
        model = bm.FFNN(input_dim=input_layer.dim, in_dropout_prob=float(config['in_dropout']),
                        hidden_size=int(config['hidden_size']),
                        add_topic=(config.get('add_resid', '1') == '1'), use_cuda=use_cuda, bias=False)
        optimizer = optim.Adam(model.parameters())

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch,
                  'batching_kwargs': batch_args, 'name': config['name'],
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn,
                  'fine_tune': (config.get('fine-tune', 'no') == 'yes')}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'], save_ckp=(args['save_ckp'] == 1),
                                                      **kwargs)





    elif 'tganet' in config['name']: # Author's model asdf
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

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch,
                  'batching_kwargs': batch_args, 'name': config['name'],# + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn}

        # -------
        # My modelling start
        print("Run the model!")
        filename='../checkpoints/ckp-tganet_t7-BEST.tar'
        checkpoint = torch.load(filename)
        #print( checkpoint )
        model.load_state_dict( checkpoint['model_state_dict'] )
        # TODO - load data, process and run in model
        # if data doesn't work use dataloader
        input_data, true_labels, id_lst = foo( dataloader ) #self.prepare_data(data, type_lst=type_lst)

        batching_fn = data_utils.prepare_batch
        samples = [i for i in dataloader]

        sample_batched = samples[0]
        print( f"BATCH ARGS: {batch_args.keys()}" )
        args = batching_fn(sample_batched, **batch_args)
        print( "args:" )
        print( args.keys() )
        print( args )
        label_tensor = torch.tensor(args["labels"])


        # EMBEDDING
        embed_args = input_layer(**args) # embed model in model_utils
        args.update(embed_args)

        #suf = setup_fn(args, False)
        # PREDICTION
        print("START PREDICTION!")
        print( "ARGS:" )
        device = 'cuda' if use_cuda else 'cpu'
        txt_E  = args['txt_E'].to(device)  # (B,E)
        top_E  = args['top_E'].to(device)  # (B,E)
        top_rep = args['avg_top_E'].to(device)
        text_l = args['txt_l'].to(device)
        print_args( args )

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
        
        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'], save_ckp=(args['save_ckp'] == 1),
                                                      **kwargs)















    elif 'repffnn' in config['name']:
        batch_args = {'keep_sen': False}
        input_layer = im.JointBERTLayerWithExtra(vecs=topic_vecs, use_cuda=use_cuda,
                                                 use_both=(config.get('use_ori_topic', '1') == '1'),
                                                 static_vecs=(config.get('static_topics', '1') == '1'))

        setup_fn = data_utils.setup_helper_bert_attffnn

        loss_fn = nn.CrossEntropyLoss()

        model = bm.RepFFNN(in_dropout_prob=float(config['in_dropout']),
                                 hidden_size=int(config['hidden_size']),
                                 input_dim=int(config['topic_dim']),
                                 use_cuda=use_cuda)

        optimizer = optim.Adam(model.parameters())

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch,
                  'batching_kwargs': batch_args, 'name': config['name'],  # + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': optimizer,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda,
                                                      checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                      result_path=config.get('res_path', 'data/gen-stance/'),
                                                      use_score=args['score_key'], save_ckp=(args['save_ckp'] == 1),
                                                      **kwargs)


    start_time = time.time()
    train(model_handler, int(config['epochs']), dev_data=dev_dataloader, early_stopping=args['early_stop'],
          num_warm=args['num_warm'], is_bert=('bert' in config))
    print("[{}] total runtime: {:.2f} minutes".format(config['name'], (time.time() - start_time)/60.))
