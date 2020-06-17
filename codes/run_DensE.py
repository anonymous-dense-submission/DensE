#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


from model import DensEModel
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-me', '--entity_embedding_has_mod', action='store_true')
    parser.add_argument('-mr', '--relation_embedding_has_mod', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=40, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=500, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--study_valid_id', type=int, default=0, help='relation_id_to_study in valid')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.entity_embedding_has_mod = argparse_dict['entity_embedding_has_mod']
    args.relation_embedding_has_mod = argparse_dict['relation_embedding_has_mod']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    args.gamma = argparse_dict['gamma']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_x_embedding = model.entity_x.weight.data.detach().cpu().numpy()
    entity_y_embedding = model.entity_y.weight.data.detach().cpu().numpy()
    entity_z_embedding = model.entity_z.weight.data.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, 'entity_x'), entity_x_embedding)
    np.save(os.path.join(args.save_path, 'entity_y'), entity_y_embedding)
    np.save(os.path.join(args.save_path, 'entity_z'), entity_z_embedding)
    
    relation_w_embedding = model.relation_w.weight.data.detach().cpu().numpy()
    relation_x_embedding = model.relation_x.weight.data.detach().cpu().numpy()
    relation_y_embedding = model.relation_y.weight.data.detach().cpu().numpy()
    relation_z_embedding = model.relation_z.weight.data.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, 'relation_w'), relation_w_embedding)
    np.save(os.path.join(args.save_path, 'relation_x'), relation_x_embedding)
    np.save(os.path.join(args.save_path, 'relation_y'), relation_y_embedding)
    np.save(os.path.join(args.save_path, 'relation_z'), relation_z_embedding)
    
def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
def make_additional_symmetric_fake_graph(train_trip, max_node_id):
    new_train_lst = []
    add_nodes_id = max_node_id
    for trip in train_trip:
        new_train_lst.append(trip)
        new_train_lst.append((trip[2] + add_nodes_id, trip[1], trip[0] + add_nodes_id))
    return new_train_lst

def reciprocal_fake_graph(input_trip, n_relation):
    new_train_lst = []
    reciprocal_lst = []
    for trip in input_trip:
        new_train_lst.append(trip)
        new_train_lst.append((trip[2], trip[1] + n_relation, trip[0]))
        reciprocal_lst.append((trip[2], trip[1] + n_relation, trip[0]))
    return new_train_lst, reciprocal_lst
        
def add_star_node(max_node_id, nentity, max_rel_id):
    new_train_lst = []
    for node in range(max_node_id):
        new_train_lst.append((node, max_rel_id, nentity))
        new_train_lst.append((nentity, max_rel_id, node))
    return new_train_lst

def main(args):
    
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id) * 2
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('Evaluating on Valid Dataset specific relation ID %d' %(args.study_valid_id))
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    _train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    _valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    _test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    _all_true_triples = _train_triples + _valid_triples + _test_triples
    logging.info('#real_train: %d' % len(_train_triples))
    logging.info('#real_valid: %d' % len(_valid_triples))
    logging.info('#real_test: %d' % len(_test_triples))
    
    
    train_triples = reciprocal_fake_graph(_train_triples, len(relation2id))
    valid_triples = reciprocal_fake_graph(_valid_triples, len(relation2id))
    test_triples = reciprocal_fake_graph(_test_triples, len(relation2id))
    logging.info('#train: %d' % len(train_triples[0]))
    logging.info('#valid: %d' % len(valid_triples[0]))
    logging.info('#test: %d' % len(test_triples[0]))
    #All true triples
    all_true_triples = train_triples[0] + valid_triples[0] + test_triples[0]
    reciprocal_all_true_triples = train_triples[1] + valid_triples[1] + test_triples[1]
    
    kge_model = DensEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        entity_embedding_has_mod=args.entity_embedding_has_mod,
        relation_embedding_has_mod=args.relation_embedding_has_mod
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples[0], all_true_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples[0], all_true_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )

        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=1000, verbose=True, factor=0.5, cooldown=500, min_lr=0.0000002)
        
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    logging.info('entity_embedding_has_mod = %s' % args.entity_embedding_has_mod)
    logging.info('relation_embedding_has_mod = %s' % str(args.relation_embedding_has_mod))
    
    
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        max_mrr = 0.
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, step, args)
            
            scheduler.step(log['loss'])
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and (step+1) % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, _valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
                
                if metrics['MRR'] > max_mrr:
                    logging.info('Better Performance on Valid, save model')
                    max_mrr = metrics['MRR']
                    
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args)

#         save_variable_list = {
#             'step': step, 
#             'current_learning_rate': current_learning_rate,
#             'warm_up_steps': warm_up_steps
#         }
#         save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, _valid_triples, _all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, _test_triples, _all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, _train_triples, _all_true_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
