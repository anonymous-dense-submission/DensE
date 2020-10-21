#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

import torch.autograd as autograd
from torch.autograd import Variable
from numpy.random import RandomState

class DensEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 entity_embedding_has_mod=False, relation_embedding_has_mod=False):
        super(DensEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 1.2
        self.rel_high_bound = 2.0
        
        self.use_abs_norm = True
        self.allow_minus_mod = True
        self.use_entity_phase = False
        self.use_real_part = False
        
        self.criterion = 'he'
        
        if self.criterion == 'glorot':
            mod_range = 1. / np.sqrt(2 * (self.hidden_dim + self.hidden_dim))
        elif self.criterion == 'he':
            mod_range = 1. / np.sqrt(2 * self.hidden_dim)
        
        if self.allow_minus_mod:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range * 2.]), 
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([mod_range]), 
                requires_grad=False
            )
            
        self.gamma1 = nn.Parameter(
            torch.Tensor([(self.rel_high_bound + self.epsilon) * mod_range * self.hidden_dim]), 
            requires_grad=False
        )
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.unit_mod = nn.Parameter(
            torch.Tensor([1.]), 
            requires_grad=False
        )
        
        self.zero_ent_phase = nn.Parameter(
            torch.Tensor([0.]), 
            requires_grad=False
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.entity_embedding_has_mod = entity_embedding_has_mod
        self.relation_embedding_has_mod = relation_embedding_has_mod
                   
        self.entity_x = nn.Embedding(self.nentity, self.hidden_dim)
        self.entity_y = nn.Embedding(self.nentity, self.hidden_dim)   
        self.entity_z = nn.Embedding(self.nentity, self.hidden_dim)
        
        self.relation_w = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_x = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_y = nn.Embedding(self.nrelation, self.hidden_dim)
        self.relation_z = nn.Embedding(self.nrelation, self.hidden_dim)
        
        self.init_weights()
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['DensE']:
            raise ValueError('model %s not supported' % model_name)
        if self.use_real_part:
            try:
                assert(self.use_abs_norm == True)
            except:
                raise ValueError('use_abs_norm should be true if you only use real part')
        if (not self.entity_embedding_has_mod) and self.relation_embedding_has_mod:
            raise ValueError('when relation has mod, entity must have mod')
                
            
    def init_weights(self):

        rel_w, rel_x, rel_y, rel_z = self.relation_init(self.nrelation, self.hidden_dim)
        rel_w, rel_x, rel_y, rel_z = torch.from_numpy(rel_w), torch.from_numpy(rel_x), torch.from_numpy(rel_y), torch.from_numpy(rel_z)
        self.relation_w.weight.data = rel_w.type_as(self.relation_w.weight.data)
        self.relation_x.weight.data = rel_x.type_as(self.relation_x.weight.data)
        self.relation_y.weight.data = rel_y.type_as(self.relation_y.weight.data)
        self.relation_z.weight.data = rel_z.type_as(self.relation_z.weight.data)
        
        e_x, e_y, e_z = self.entity_init(self.nentity, self.hidden_dim)
        e_x, e_y, e_z = torch.from_numpy(e_x), torch.from_numpy(e_y), torch.from_numpy(e_z)
        self.entity_x.weight.data = e_x.type_as(self.entity_x.weight.data)
        self.entity_y.weight.data = e_y.type_as(self.entity_y.weight.data)
        self.entity_z.weight.data = e_z.type_as(self.entity_z.weight.data)
        
#     def relation_init(self, n_entries, features, criterion='he'):
#         fan_in = features
#         fan_out = features
        
#         if criterion == 'glorot':
#             s = 1. / np.sqrt(2 * (fan_in + fan_out))
#         elif criterion == 'he':
#             s = 1. / np.sqrt(2 * fan_in)
#         else:
#             raise ValueError('Invalid criterion: ', criterion)
            
#         print('INFO: init rel_mod is: ', s)

#         kernel_shape = (n_entries, features)
            
#         rel_mod = np.random.uniform(low=-s, high=s, size=kernel_shape)
#         rotate_phase = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=kernel_shape)
#         theta = np.random.uniform(low=0, high=np.pi, size=kernel_shape)
#         phi = np.random.uniform(low=0, high=2*np.pi, size=kernel_shape)
        
#         rel_w = rel_mod * np.cos(rotate_phase/2)
#         rel_x = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.cos(phi)
#         rel_y = rel_mod * np.sin(rotate_phase/2) * np.sin(theta) * np.sin(phi)
#         rel_z = rel_mod * np.sin(rotate_phase/2) * np.cos(theta)

#         return rel_w, rel_x, rel_y, rel_z

    def relation_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features
        
        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        
        print('INFO: init rel_mod is: ', s)

        kernel_shape = (n_entries, features)
        
        rel_w = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rel_x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rel_y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        rel_z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return rel_w, rel_x, rel_y, rel_z
    
    def entity_init(self, n_entries, features, criterion='he'):
        fan_in = features
        fan_out = features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
            
        print('INFO: init x, y, z is: ', s)

        # rng = RandomState(456)
        kernel_shape = (n_entries, features)
            
        x = np.random.uniform(low=-s, high=s, size=kernel_shape)
        y = np.random.uniform(low=-s, high=s, size=kernel_shape)
        z = np.random.uniform(low=-s, high=s, size=kernel_shape)

        return x, y, z
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # batch_size, negative_sample_size = sample.size(0), 1
            
            head_x = self.entity_x(sample[:, 0]).unsqueeze(1)
            head_y = self.entity_y(sample[:, 0]).unsqueeze(1)
            head_z = self.entity_z(sample[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(sample[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(sample[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(sample[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(sample[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(sample[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(sample[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(sample[:, 1]).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_x = self.entity_x(head_part)
            head_y = self.entity_y(head_part)
            head_z = self.entity_z(head_part)
            
            tail_x = self.entity_x(tail_part[:, 2]).unsqueeze(1)
            tail_y = self.entity_y(tail_part[:, 2]).unsqueeze(1)
            tail_z = self.entity_z(tail_part[:, 2]).unsqueeze(1)
            
            rel_w = self.relation_w(tail_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(tail_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(tail_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(tail_part[:, 1]).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_x = self.entity_x(head_part[:, 0]).unsqueeze(1)
            head_y = self.entity_y(head_part[:, 0]).unsqueeze(1)
            head_z = self.entity_z(head_part[:, 0]).unsqueeze(1)
            
            tail_x = self.entity_x(tail_part)
            tail_y = self.entity_y(tail_part)
            tail_z = self.entity_z(tail_part)
            
            rel_w = self.relation_w(head_part[:, 1]).unsqueeze(1)
            rel_x = self.relation_x(head_part[:, 1]).unsqueeze(1)
            rel_y = self.relation_y(head_part[:, 1]).unsqueeze(1)
            rel_z = self.relation_z(head_part[:, 1]).unsqueeze(1)
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'DensE': self.DensE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_x, head_y, head_z, 
                                                rel_w, rel_x, rel_y, rel_z, 
                                                tail_x, tail_y, tail_z, 
                                                mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def DensE(self, head_x, head_y, head_z, 
                   rel_w, rel_x, rel_y, rel_z, 
                   tail_x, tail_y, tail_z, 
                   mode):
        pi = 3.14159265358979323846
        assert(self.use_entity_phase == False)
        assert(self.use_real_part == False)
        
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
        
        compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
        compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
        compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
        
        if self.relation_embedding_has_mod:
            compute_tail_x = denominator * compute_tail_x
            compute_tail_y = denominator * compute_tail_y
            compute_tail_z = denominator * compute_tail_z
        
        delta_x = (compute_tail_x - tail_x)
        delta_y = (compute_tail_y - tail_y)
        delta_z = (compute_tail_z - tail_z)
        
        score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
        score1 = score1.norm(dim = 0)
        
        x = -x
        y = -y
        z = -z
        compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
        compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
        compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
        
        if self.relation_embedding_has_mod:
            compute_head_x = compute_head_x / denominator
            compute_head_y = compute_head_y / denominator
            compute_head_z = compute_head_z / denominator
        
        delta_x2 = (compute_head_x - head_x)
        delta_y2 = (compute_head_y - head_y)
        delta_z2 = (compute_head_z - head_z)
        
        score2 = torch.stack([delta_x2, delta_y2, delta_z2], dim = 0)
        score2 = score2.norm(dim = 0)     
        
        score1 = score1.mean(dim=2)
        score2 = score2.mean(dim=2)

#         score1 = score1.sum(dim=2)
#         score2 = score2.sum(dim=2)
        
        score = (score1 + score2) / 2
        
        score = self.gamma.item() - score
            
        return score, score1, score2, torch.abs(delta_x)

    @staticmethod
    def train_step(model, optimizer, train_iterator, step, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode=mode) # 全是负样本分数 shape: batch_size, neg_size
        
        if step % 500 == 0:
            print(negative_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, head_mod, tail_mod, rel_mod = model(positive_sample) # 正样本分数 shape: batch_size, 1     

        if step % 500 == 0:
            print(positive_score.mean(), head_mod.mean(), tail_mod.mean(), rel_mod.mean())

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_x.weight.data.norm(p = 3)**3 + 
                model.entity_y.weight.data.norm(p = 3)**3 + 
                model.entity_z.weight.data.norm(p = 3)**3 
            ) / args.batch_size

            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
#             'train_hit1': train_hit1
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation/2, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score, head_mod, tail_mod, rel_mod = model((positive_sample, negative_sample), mode)
#                         print(filter_bias, filter_bias.shape, filter_bias.sum())
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
    
    