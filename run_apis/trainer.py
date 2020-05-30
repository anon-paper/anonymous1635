import logging
import time

import torch.nn as nn

from dataset.prefetch_data import data_prefetcher
from tools import utils


class Trainer(object):
    def __init__(self, train_data, val_data, optimizer=None, criterion=None, 
                    scheduler=None, config=None, report_freq=None):
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.report_freq = report_freq
    
    def train(self, model, epoch, model_train=True):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        model.train() if model_train else model.eval()
        start = time.time()

        prefetcher = data_prefetcher(self.train_data)
        input, target = prefetcher.next()
        step = 0
        while input is not None:        
            data_t = time.time() - start
            
            if step == 0:
                self.scheduler.step() # epoch=epoch
            elif self.config.optim.lr_schedule == 'cosine':
                self.scheduler.step()
                
            n = input.size(0)
            if step==0:
                logging.info('epoch %d lr %e', epoch, self.optimizer.param_groups[0]['lr'])
            self.optimizer.zero_grad()
            
            logits= model(input)
            if self.config.optim.label_smooth:
                loss = self.criterion(logits, target, self.config.optim.smooth_alpha)
            else:
                loss = self.criterion(logits, target)

            loss.backward()
            if self.config.optim.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip)
            self.optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        
            batch_t = time.time() - start
            start = time.time()

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)
            if step!=0 and step % self.report_freq == 0:
                logging.info(
                    'Train epoch %03d step %03d | loss %.4f  top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f', 
                    epoch, step, objs.avg, top1.avg, top5.avg, batch_time.avg, data_time.avg)
            input, target = prefetcher.next()
            step += 1
        logging.info('EPOCH%d Train_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', 
                                epoch, top1.avg, top5.avg, batch_time.avg, data_time.avg)

        return top1.avg, top5.avg, objs.avg, batch_time.avg, data_time.avg

    def infer(self, model, epoch=0):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        model.eval()

        start = time.time()
        prefetcher = data_prefetcher(self.val_data)
        input, target = prefetcher.next()
        step = 0
        while input is not None:
            step += 1
            data_t = time.time() - start
            n = input.size(0)

            logits = model(input)
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            batch_t = time.time() - start
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)

            if step % self.report_freq == 0:
                logging.info(
                    'Val epoch %03d step %03d | top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f', 
                    epoch, step, top1.avg, top5.avg, batch_time.avg, data_time.avg)
            start = time.time()
            input, target = prefetcher.next()

        logging.info('EPOCH%d Valid_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', 
                                epoch, top1.avg, top5.avg, batch_time.avg, data_time.avg)
        return top1.avg, top5.avg, batch_time.avg, data_time.avg
