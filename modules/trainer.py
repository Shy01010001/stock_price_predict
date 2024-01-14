# -*- coding: utf-8 -*-
import os
import logging
from abc import abstractmethod

from numpy import inf
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from modules.base_cmn import set_flag
from modules.metrics import calculate_accuracy
import pdb



class BaseTrainer(object):
    def __init__(self, model, optimizer, args):
        
        
        self.args = args
        self.experiment_name = args.experiment_name
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.optimizer = optimizer


        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)
            # self._record_best(log)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save the best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode=='min' and log[self.mnt_metric]<=self.mnt_best) or \
                                (self.mnt_mode=='max' and log[self.mnt_metric]>=self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled."
                            .format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops."
                                      .format(self.early_stop))
                    # break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        # improved_valid = (self.mnt_mode=='min' and log[self.mnt_metric]<=self.best_recorder['val'][self.mnt_metric]) or \
        #                  (self.mnt_mode=='max' and log[self.mnt_metric]>=self.best_recorder['val'][self.mnt_metric])
        # if improved_valid:
        #     self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode=='min' and log[self.mnt_metric_test]<=self.best_recorder['test'][self.mnt_metric_test]) or \
                        (self.mnt_mode=='max' and log[self.mnt_metric_test]>=self.best_recorder['test'][self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        print('n_gpu:',n_gpu)
        if n_gpu_use>0 and n_gpu==0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine."
                    .format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use>0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_%d.pth'%epoch)
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {}...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {}...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, args,
                 train_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, optimizer, args)
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        try:
            os.mkdir(f'./loss/{self.experiment_name}')
        except:
            print('experiment record already exist')
            pass
        self.writer = SummaryWriter(log_dir=f'./loss/{self.experiment_name}', flush_secs=20)
        self.writer1 = SummaryWriter(log_dir='./evaluation', flush_secs=20)
    def _train_epoch(self, epoch):
        log = {}
        # if (epoch - 1) % 4 == 0:
        # self._resume_checkpoint(f'./results/iu_xray/checkpoint_epoch_{epoch}.pth')
        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        
        self.model.train()
        # pdb.set_trace()
        for batch_idx, (input_data, label) in enumerate(self.train_dataloader):
        # tuple: len=batch_size "CXR2384_IM-0942"等, [batch_size, image_num, 3, 224, 224]
        # [batch_size, max_seq_len], [batch_size, max_seq_len]
            # break    
            # print('reports_', reports_ids[0])
            # inp = torch.rand(4, 30, 4096).cuda()
            input_data = input_data.cuda()
            label = label.cuda()
            input_data = input_data[:, :, :-1].to(dtype=torch.float32)
            label = label[:, :-1].to(dtype=torch.float32)
            # print(input_data.size())
            # print(label.size())
            # exit()
            output = self.model(input_data) # [batch_size, max_seq_len-1, vocab_size+1]
            output = output[:, -1, :]
            # print(output.size())
      
            loss = self.criterion(output, label) # 6~7
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                log = {'train_loss': train_loss / len(self.train_dataloader)}
        
        self.writer.add_scalar('Loss/Train', train_loss / len(self.train_dataloader), epoch)
        self.writer.add_scalar('LR/Train', self.optimizer.param_groups[0]['lr'] , epoch)
        ###################################################     test    #####################################################################

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        score = 0
        with torch.no_grad():
            set_flag.set_epoch(epoch)
            for batch_idx, (input_data, label) in enumerate(self.test_dataloader):
                input_data = input_data.cuda()
                label = label.cuda()
                input_data = input_data[:, :, :-1].to(dtype=torch.float32)
                label = label[:, :-1].to(dtype=torch.float32)
                # print(input_data.size())
                # print(label.size())
                # exit()
                output = self.model(input_data) # [batch_size, max_seq_len-1, vocab_size+1]
                output = output[:, -1, :]
                # print(output.size())
          
                loss = self.criterion(output, label) # 6~7
                train_loss += loss.item()
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                print(output)
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n***********************************\n!!!!!!!!!!!!!!!!!')
                print(label)
                exit()
                score = score + calculate_accuracy(output, label) 
                # log = {'train_loss': train_loss / len(self.train_dataloader)}
                    # print(expert)
        score = score / len(self.test_dataloader)
        print(f'epoch {epoch} accurency: \n', score)
        self.writer.add_scalar('Currency', score, epoch)
        score = 0
        # else:
        #     pass
        return log

