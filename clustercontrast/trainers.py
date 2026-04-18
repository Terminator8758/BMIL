from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import numpy as np



class ClusterContrastTrainer_Stage1(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_Stage1, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory

    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=200):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = self._forward(inputs_rgb, inputs_ir,
                                                                                             label_1=labels_rgb,
                                                                                             label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir + loss_rgb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)




class ClusterContrastTrainer_Stage2(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_Stage2, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory

    def train(self, epoch, data_loader_ir, data_loader_rgb, data_loader_all_ir, data_loader_all_rgb,
              optimizer, print_freq=10, train_iters=200, has_global_cluster_loss=True):

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, _, _ = self._forward(inputs_rgb, inputs_ir, label_1=labels_rgb, label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            loss = loss_ir + loss_rgb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # -------------------------------------------------------------------------
            ### global loss
            loss2 = torch.tensor(0.)
            
            if has_global_cluster_loss:
                inputs_all_ir = data_loader_all_ir.next()
                inputs_all_rgb = data_loader_all_rgb.next()
                # process inputs
                inputs_all_ir, labels_all_ir, indexes_all_ir = self._parse_data_ir(inputs_all_ir)
                inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = self._parse_data_rgb(inputs_all_rgb)
                # forward
                inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
                labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)

                _, f_out_all_rgb, f_out_all_ir, _, _, _, _ = self._forward(inputs_all_rgb, inputs_all_ir,
                                                                        label_1=labels_all_rgb, label_2=labels_all_ir, modal=0)

                # when using modality-specific prototypes:
                loss_all_ir = self.memory_all(f_out_all_ir, indexes_all_ir)  # note: indexes here are proxy index
                loss_all_rgb = self.memory_all(f_out_all_rgb, torch.cat((indexes_all_rgb, indexes_all_rgb)))
                loss2 = 0.5*(loss_all_ir + loss_all_rgb)
            
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()
            # ----------------------------------------------------------------------------

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss global {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb, loss2))


    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        if len(inputs) == 6:
            imgs, _, _, pids, _, indexes = inputs
        else:
            imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)







