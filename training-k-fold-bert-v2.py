# import os and define graphic card
import os

os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import random
import argparse
import pandas as pd
import numpy as np

# import pytorch related libraries
import torch
from tensorboardX import SummaryWriter
from transformers import *

# import apex for mix precision training
from apex import amp

amp.register_half_function(torch, "einsum")
from apex.optimizers import FusedAdam

# import dataset class
from dataset.dataset_v2 import *

# import utils
from utils.squad_metrics import *
from utils.ranger import *
from utils.lrs_scheduler import *
from utils.loss_function import *
from utils.metric import *
from utils.file import *

# import model
from model.model_bert import *

# import config
from config_bert import *

############################################################################## Define Argument
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--fold', type=int, default=0, required=False, help="specify the fold for training")
parser.add_argument('--model_type', type=str, default="roberta-base", required=False, help="specify the model type")
parser.add_argument('--seed', type=int, default=2020, required=False, help="specify the seed")
parser.add_argument('--batch_size', type=int, default=16, required=False, help="specify the batch size")
parser.add_argument('--accumulation_steps', type=int, default=1, required=False, help="specify the accumulation_steps")
parser.add_argument('--Datasampler', type=str, default="ImbalancedDatasetSampler", required=False, help="specify the datasampler")


############################################################################## seed All
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['PYHTONHASHseed'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


############################################################################## Class for QA
class QA():
    def __init__(self, config):
        super(QA).__init__()
        self.config = config
        self.setup_logger()
        self.setup_gpu()
        self.load_data()
        self.prepare_train()
        self.setup_model()

    def setup_logger(self):
        self.log = Logger()
        self.log.open((os.path.join(self.config.checkpoint_folder, "train_log.txt")), mode='a+')

    def setup_gpu(self):
        # confirm the device which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.num_device = torch.cuda.device_count()
        if self.config.use_gpu:
            self.config.device = 'cuda'
            if self.num_device <= 1:
                self.config.data_parallel = False
            elif self.config.data_parallel:
                torch.multiprocessing.set_start_method('spawn', force=True)
        else:
            self.config.device = 'cpu'
            self.config.data_parallel = False

    def load_data(self):
        self.log.write('\nLoading data...')

        get_train_val_split(data_path=self.config.data_path,
                            save_path=self.config.save_path,
                            n_splits=self.config.n_splits,
                            seed=self.config.seed,
                            split=self.config.split)

        self.test_data_loader, self.tokenizer = get_test_loader(data_path=self.config.data_path,
                                                max_seq_length=self.config.max_seq_length,
                                                model_type=self.config.model_type,
                                                batch_size=self.config.val_batch_size,
                                                num_workers=self.config.num_workers)

        self.train_data_loader, self.val_data_loader, _ = get_train_val_loaders(data_path=self.config.data_path,
                                                                  seed=self.config.seed,
                                                                  fold=self.config.fold,
                                                                  max_seq_length=self.config.max_seq_length,
                                                                  model_type=self.config.model_type,
                                                                  batch_size=self.config.batch_size,
                                                                  val_batch_size=self.config.val_batch_size,
                                                                  num_workers=self.config.num_workers,
                                                                  Datasampler=self.config.Datasampler)

    def prepare_train(self):
        # preparation for training
        self.step = 0
        self.epoch = 0
        self.finished = False
        self.valid_epoch = 0
        self.train_loss, self.valid_loss, self.valid_metric_optimal = float('-inf'), float('-inf'), float('-inf')
        self.writer = SummaryWriter()
        ############################################################################### eval setting
        self.eval_step = int(len(self.train_data_loader) * self.config.saving_rate)
        self.log_step = int(len(self.train_data_loader) * self.config.progress_rate)
        self.eval_count = 0
        self.count = 0

    def pick_model(self):
        # for switching model
        self.model = TweetBert(model_type=self.config.model_type,
                               hidden_layers=self.config.hidden_layers).to(self.config.device)

        def init_weights(model):
            for name, param in model.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param.data, std=0.02)

        self.model.cross_attention.apply(init_weights)
        self.model.cross_attention.apply(init_weights)
        self.model.qa_start.apply(init_weights)
        self.model.qa_end.apply(init_weights)
        self.model.qa_classifier.apply(init_weights)

        if self.config.load_pretrain:
            checkpoint_to_load = torch.load(self.config.checkpoint_pretrain, map_location=self.config.device)
            model_state_dict = checkpoint_to_load

            if self.config.data_parallel:
                state_dict = self.model.model.state_dict()
            else:
                state_dict = self.model.state_dict()

            keys = list(state_dict.keys())

            for key in keys:
                if any(s in key for s in (self.config.skip_layers + ["qa_classifier.weight", "qa_classifier.bias"])):
                    continue
                try:
                    state_dict[key] = model_state_dict[key]
                except:
                    print("Missing key:", key)

            if self.config.data_parallel:
                self.model.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)


    def differential_lr(self):

        if self.config.differential_lr:
            self.optimizer_grouped_parameters = []
            list_lr = []

            if ((self.config.model_type == "bert-base-uncased") or (self.config.model_type == "bert-base-cased")
                    or self.config.model_type == "roberta-base"):

                list_layers = [self.model.bert.embeddings,
                               self.model.bert.encoder.layer[0],
                               self.model.bert.encoder.layer[1],
                               self.model.bert.encoder.layer[2],
                               self.model.bert.encoder.layer[3],
                               self.model.bert.encoder.layer[4],
                               self.model.bert.encoder.layer[5],
                               self.model.bert.encoder.layer[6],
                               self.model.bert.encoder.layer[7],
                               self.model.bert.encoder.layer[8],
                               self.model.bert.encoder.layer[9],
                               self.model.bert.encoder.layer[10],
                               self.model.bert.encoder.layer[11],
                               self.model.aoa,
                               self.model.cross_attention,
                               self.model.qa_start,
                               self.model.qa_end,
                               self.model.qa_classifier,
                               ]

            elif ((self.config.model_type == "bert-large-uncased") or (self.config.model_type == "bert-large-cased")
                  or self.config.model_type == "roberta-large"):

                list_layers = [self.model.bert.embeddings,
                               self.model.bert.encoder.layer[0],
                               self.model.bert.encoder.layer[1],
                               self.model.bert.encoder.layer[2],
                               self.model.bert.encoder.layer[3],
                               self.model.bert.encoder.layer[4],
                               self.model.bert.encoder.layer[5],
                               self.model.bert.encoder.layer[6],
                               self.model.bert.encoder.layer[7],
                               self.model.bert.encoder.layer[8],
                               self.model.bert.encoder.layer[9],
                               self.model.bert.encoder.layer[10],
                               self.model.bert.encoder.layer[11],
                               self.model.bert.encoder.layer[12],
                               self.model.bert.encoder.layer[13],
                               self.model.bert.encoder.layer[14],
                               self.model.bert.encoder.layer[15],
                               self.model.bert.encoder.layer[16],
                               self.model.bert.encoder.layer[17],
                               self.model.bert.encoder.layer[18],
                               self.model.bert.encoder.layer[19],
                               self.model.bert.encoder.layer[20],
                               self.model.bert.encoder.layer[21],
                               self.model.bert.encoder.layer[22],
                               self.model.bert.encoder.layer[23],
                               self.model.aoa,
                               self.model.cross_attention,
                               self.model.qa_start,
                               self.model.qa_end,
                               self.model.qa_classifier,
                               ]
            else:
                raise NotImplementedError

            if self.config.method == "step":
                mult = self.config.lr / self.config.min_lr
                step = mult ** (1 / (len(list_layers) - 1))
                list_lr = [self.config.min_lr * (step ** i) for i in range(len(list_layers))]
            elif self.config.method == "decay":

                for i in range(len(list_layers)):
                    list_lr.append(self.config.lr)
                    self.config.lr = self.config.lr * self.config.decay_factor
                list_lr.reverse()

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            print(list_lr)

            for i in range(len(list_lr)):

                if isinstance(list_layers[i], list):

                    for list_layer in list_layers[i]:
                        layer_parameters = list(list_layer.named_parameters())

                        self.optimizer_grouped_parameters.append({
                            'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)],
                            'lr': list_lr[i],
                            'weight_decay': self.config.weight_decay})

                        self.optimizer_grouped_parameters.append({
                            'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)],
                            'lr': list_lr[i],
                            'weight_decay': 0.0})

                else:

                    layer_parameters = list(list_layers[i].named_parameters())

                    self.optimizer_grouped_parameters.append({
                        'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)],
                        'lr': list_lr[i],
                        'weight_decay': self.config.weight_decay})

                    self.optimizer_grouped_parameters.append({
                        'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)],
                        'lr': list_lr[i],
                        'weight_decay': 0.0})

        else:
            param_optimizer = list(self.model.named_parameters())

            prefix = "bert"
            def is_backbone(n):
                return prefix in n

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            self.optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and is_backbone(n)],
                 'lr': self.config.min_lr,
                 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not is_backbone(n)],
                 'lr': self.config.lr,
                 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and is_backbone(n)],
                 'lr': self.config.min_lr,
                 'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not is_backbone(n)],
                 'lr': self.config.lr,
                 'weight_decay': 0.0}
            ]

    def prepare_optimizer(self):

        # differential lr for each sub module first
        self.differential_lr()

        # optimizer
        if self.config.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.optimizer_grouped_parameters, eps=self.config.adam_epsilon)
        elif self.config.optimizer_name == "Ranger":
            self.optimizer = Ranger(self.optimizer_grouped_parameters)
        elif self.config.optimizer_name == "AdamW":
            self.optimizer = AdamW(self.optimizer_grouped_parameters, eps=self.config.adam_epsilon)
        elif self.config.optimizer_name == "FusedAdam":
            self.optimizer = FusedAdam(self.optimizer_grouped_parameters,
                                       bias_correction=False)
        else:
            raise NotImplementedError

        # lr scheduler
        if self.config.lr_scheduler_name == "WarmupCosineAnealing":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmRestart":
            self.scheduler = WarmRestart(self.optimizer, T_max=5, T_mult=1, eta_min=1e-6)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupLinear":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6,
                                                                        patience=1, min_lr=1e-7)
            self.lr_scheduler_each_iter = False
        else:
            raise NotImplementedError

        # lr scheduler step for checkpoints
        if self.lr_scheduler_each_iter:
            self.scheduler.step(self.step)
        else:
            self.scheduler.step(self.epoch)

    def prepare_apex(self):
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def load_check_point(self):
        self.log.write('Model loaded as {}.'.format(self.config.load_point))
        checkpoint_to_load = torch.load(self.config.load_point, map_location=self.config.device)
        self.step = checkpoint_to_load['step']
        self.epoch = checkpoint_to_load['epoch']

        model_state_dict = checkpoint_to_load['model']
        if self.config.load_from_load_from_data_parallel:
            # model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
            # "module.model"
            model_state_dict = {k[13:]: v for k, v in model_state_dict.items()}

        if self.config.data_parallel:
            state_dict = self.model.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        keys = list(state_dict.keys())

        for key in keys:
            if any(s in key for s in self.config.skip_layers):
                continue
            try:
                state_dict[key] = model_state_dict[key]
            except:
                print("Missing key:", key)

        if self.config.data_parallel:
            self.model.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        if self.config.load_optimizer:
            self.optimizer.load_state_dict(checkpoint_to_load['optimizer'])

    def save_check_point(self):
        # save model, optimizer, and everything required to keep
        checkpoint_to_save = {
            'step': self.step,
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}

        save_path = self.config.save_point.format(self.step, self.epoch)
        torch.save(checkpoint_to_save, save_path)
        self.log.write('Model saved as {}.'.format(save_path))

    def setup_model(self):
        self.pick_model()

        if self.config.data_parallel:
            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

            if self.config.reuse_model:
                self.load_check_point()

            self.model = torch.nn.DataParallel(self.model)

        else:
            if self.config.reuse_model:
                self.load_check_point()

            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

    def count_parameters(self):
        # get total size of trainable parameters
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def show_info(self):
        # show general information before training
        self.log.write('\n*General Setting*')
        self.log.write('\nseed: {}'.format(self.config.seed))
        self.log.write('\nmodel: {}'.format(self.config.model_name))
        self.log.write('\ntrainable parameters:{:,.0f}'.format(self.count_parameters()))
        self.log.write("\nmodel's state_dict:")
        self.log.write('\ndevice: {}'.format(self.config.device))
        self.log.write('\nuse gpu: {}'.format(self.config.use_gpu))
        self.log.write('\ndevice num: {}'.format(self.num_device))
        self.log.write('\noptimizer: {}'.format(self.optimizer))
        self.log.write('\nreuse model: {}'.format(self.config.reuse_model))
        self.log.write('\nadversarial training: {}'.format(self.config.adversarial))
        if self.config.reuse_model:
            self.log.write('\nModel restored from {}.'.format(self.config.load_point))
        self.log.write('\n')

    def train_op(self):
        self.show_info()
        self.log.write('** start training here! **\n')
        self.log.write('   batch_size=%d,  accumulation_steps=%d\n' % (self.config.batch_size,
                                                                       self.config.accumulation_steps))
        self.log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))

        while self.epoch <= self.config.num_epoch:

            self.train_metrics = []
            self.train_metrics_no_postprocessing = []
            self.train_metrics_postprocessing = []

            # update lr and start from start_epoch
            if (self.epoch >= 1) and (not self.lr_scheduler_each_iter) \
                    and (self.config.lr_scheduler_name != "ReduceLROnPlateau"):
                self.scheduler.step()

            self.log.write("Epoch%s\n" % self.epoch)
            self.log.write('\n')

            sum_train_loss = np.zeros_like(self.train_loss)
            sum_train = np.zeros_like(self.train_loss)

            # init optimizer
            torch.cuda.empty_cache()
            self.model.zero_grad()

            for tr_batch_i, (
                    all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions,
                    all_onthot_ans_type, all_orig_tweet, all_orig_selected, all_sentiment, ) in \
                    enumerate(self.train_data_loader):

                rate = 0
                for param_group in self.optimizer.param_groups:
                    rate += param_group['lr'] / len(self.optimizer.param_groups)

                # set model training mode
                self.model.train()

                # set input to cuda mode
                all_input_ids = all_input_ids.cuda()
                all_attention_masks = all_attention_masks.cuda()
                all_token_type_ids = all_token_type_ids.cuda()
                all_start_positions = all_start_positions.cuda()
                all_end_positions = all_end_positions.cuda()
                all_onthot_ans_type = all_onthot_ans_type.cuda()

                sentiment = all_sentiment
                sentiment_weight = np.array([self.config.sentiment_weight_map[sentiment_] for sentiment_ in sentiment])
                sentiment_weight = torch.tensor(sentiment_weight).float().cuda()

                outputs = self.model(input_ids=all_input_ids, attention_mask=all_attention_masks,
                                         token_type_ids=all_token_type_ids, start_positions=all_start_positions,
                                         end_positions=all_end_positions, onthot_ans_type=all_onthot_ans_type,
                                     sentiment_weight=sentiment_weight)

                loss, start_logits, end_logits = outputs[0], outputs[1], outputs[2]

                # use apex
                if self.config.apex:
                    with amp.scale_loss(loss / self.config.accumulation_steps, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # adversarial training
                if self.config.adversarial:
                    try:
                        with torch.autograd.detect_anomaly():
                            self.model.attack()
                            outputs_adv = self.model(input_ids=all_input_ids, attention_mask=all_attention_masks,
                                                     token_type_ids=all_token_type_ids, start_positions=all_start_positions,
                                                     end_positions=all_end_positions, onthot_ans_type=all_onthot_ans_type,
                                                 sentiment_weight=sentiment_weight)
                            loss_adv = outputs_adv[0]

                            # use apex
                            if self.config.apex:
                                with amp.scale_loss(loss_adv / self.config.accumulation_steps, self.optimizer) as scaled_loss_adv:
                                    scaled_loss_adv.backward()
                                    self.model.restore()
                            else:
                                loss_adv.backward()
                                self.model.restore()
                    except:
                        print("NAN LOSS")

                if ((tr_batch_i + 1) % self.config.accumulation_steps == 0):
                    if self.config.apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.model.zero_grad()
                    # adjust lr
                    if (self.lr_scheduler_each_iter):
                        self.scheduler.step()

                    self.writer.add_scalar('train_loss_' + str(self.config.fold), loss.item(),
                                           (self.epoch - 1) * len(
                                               self.train_data_loader) * self.config.batch_size + tr_batch_i *
                                           self.config.batch_size)
                    self.step += 1

                # translate to predictions
                start_logits = torch.softmax(start_logits, dim=-1)
                end_logits = torch.softmax(end_logits, dim=-1)
                start_logits = start_logits.argmax(dim=-1)
                end_logits = end_logits.argmax(dim=-1)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                start_logits = to_numpy(start_logits)
                end_logits = to_numpy((end_logits))

                for px, tweet in enumerate(all_orig_tweet):
                    selected_tweet = all_orig_selected[px]
                    jaccard_score, final_text = calculate_jaccard_score(
                        original_tweet=tweet,
                        target_string=selected_tweet,
                        idx_start=start_logits[px],
                        idx_end=end_logits[px],
                        tokenizer=self.tokenizer,
                    )

                    if (sentiment[px] == "neutral" or len(all_orig_tweet[px].split()) < 3):
                        self.train_metrics_postprocessing.append(jaccard(tweet.strip(), selected_tweet.strip()))
                        self.train_metrics.append(jaccard(tweet.strip(), selected_tweet.strip()))
                    else:
                        self.train_metrics_no_postprocessing.append(jaccard_score)
                        self.train_metrics.append(jaccard_score)

                l = np.array([loss.item() * self.config.batch_size])
                n = np.array([self.config.batch_size])
                sum_train_loss = sum_train_loss + l
                sum_train = sum_train + n

                # log for training
                if (tr_batch_i + 1) % self.log_step == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...] = 0
                    mean_train_metric = np.mean(self.train_metrics)
                    mean_train_metric_postprocessing = np.mean(self.train_metrics_postprocessing)
                    mean_train_metric_no_postprocessing = np.mean(self.train_metrics_no_postprocessing)

                    self.log.write('lr: %f train loss: %f train_jaccard: %f train_jaccard_postprocessing: %f train_jaccard_no_postprocessing: %f\n' % \
                                   (rate, train_loss[0], mean_train_metric, mean_train_metric_postprocessing, mean_train_metric_no_postprocessing))

                    print("Training ground truth: ", selected_tweet)
                    print("Training prediction: ", final_text)

                if (tr_batch_i + 1) % self.eval_step == 0:
                    self.evaluate_op()

            if self.count >= self.config.early_stopping:
                break

            self.epoch += 1

    def evaluate_op(self):

        self.eval_count += 1
        valid_loss = np.zeros(1, np.float32)
        valid_num = np.zeros_like(valid_loss)

        self.eval_metrics = []
        self.eval_metrics_no_postprocessing = []
        self.eval_metrics_postprocessing = []

        all_result = []

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()

            for val_batch_i, (
                    all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions,
                    all_onthot_ans_type, all_orig_tweet, all_orig_selected, all_sentiment, ) in \
                    enumerate(self.val_data_loader):

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                all_input_ids = all_input_ids.cuda()
                all_attention_masks = all_attention_masks.cuda()
                all_token_type_ids = all_token_type_ids.cuda()
                all_start_positions = all_start_positions.cuda()
                all_end_positions = all_end_positions.cuda()
                all_onthot_ans_type = all_onthot_ans_type.cuda()
                sentiment = all_sentiment

                outputs = self.model(input_ids=all_input_ids, attention_mask=all_attention_masks,
                                         token_type_ids=all_token_type_ids, start_positions=all_start_positions,
                                         end_positions=all_end_positions, onthot_ans_type=all_onthot_ans_type)

                loss, start_logits, end_logits = outputs[0], outputs[1], outputs[2]

                self.writer.add_scalar('val_loss_' + str(self.config.fold), loss.item(), (self.eval_count - 1) * len(
                    self.val_data_loader) * self.config.val_batch_size + val_batch_i * self.config.val_batch_size)

                # translate to predictions
                start_logits = torch.softmax(start_logits, dim=-1)
                end_logits = torch.softmax(end_logits, dim=-1)
                start_logits = start_logits.argmax(dim=-1)
                end_logits = end_logits.argmax(dim=-1)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                start_logits = to_numpy(start_logits)
                end_logits = to_numpy((end_logits))

                for px, tweet in enumerate(all_orig_tweet):
                    selected_tweet = all_orig_selected[px]
                    jaccard_score, final_text = calculate_jaccard_score(
                        original_tweet=tweet,
                        target_string=selected_tweet,
                        idx_start=start_logits[px],
                        idx_end=end_logits[px],
                        tokenizer=self.tokenizer,
                    )

                    all_result.append(final_text)

                    if (sentiment[px] == "neutral" or len(all_orig_tweet[px].split()) < 3):
                        self.eval_metrics_postprocessing.append(jaccard(tweet.strip(), selected_tweet.strip()))
                        self.eval_metrics.append(jaccard(tweet.strip(), selected_tweet.strip()))
                    else:
                        self.eval_metrics_no_postprocessing.append(jaccard_score)
                        self.eval_metrics.append(jaccard_score)

                l = np.array([loss.item() * self.config.val_batch_size])
                n = np.array([self.config.val_batch_size])
                valid_loss = valid_loss + l
                valid_num = valid_num + n

            valid_loss = valid_loss / valid_num
            mean_eval_metric = np.mean(self.eval_metrics)
            mean_eval_metric_postprocessing = np.mean(self.eval_metrics_postprocessing)
            mean_eval_metric_no_postprocessing = np.mean(self.eval_metrics_no_postprocessing)

            self.log.write('validation loss: %f eval_jaccard: %f eval_jaccard_postprocessing: %f eval_jaccard_no_postprocessing: %f\n' % \
                           (valid_loss[0], mean_eval_metric, mean_eval_metric_postprocessing, mean_eval_metric_no_postprocessing))
            print("Validating ground truth: ", selected_tweet)
            print("Validating prediction: ", final_text)

        if self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(mean_eval_metric)

        if (mean_eval_metric >= self.valid_metric_optimal):

            self.log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(
                self.valid_metric_optimal, mean_eval_metric))

            self.valid_metric_optimal = mean_eval_metric
            self.save_check_point()

            self.count = 0

        else:
            self.count += 1

        val_df = pd.DataFrame({'selected_text': all_result})
        val_df.to_csv(os.path.join(self.config.checkpoint_folder, "val_prediction_{}_{}.csv".format(self.config.seed,
                                                                                                    self.config.fold)), index=False)

    def infer_op(self):

        all_results = []

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()

            for test_batch_i, (all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions,
                               all_end_positions, all_onthot_ans_type, all_orig_tweet, all_orig_selected, all_sentiment,
                               ) in enumerate(self.test_data_loader):

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                all_input_ids = all_input_ids.cuda()
                all_attention_masks = all_attention_masks.cuda()
                all_token_type_ids = all_token_type_ids.cuda()
                all_onthot_ans_type = all_onthot_ans_type.cuda()
                sentiment = all_sentiment

                outputs = self.model(input_ids=all_input_ids, attention_mask=all_attention_masks,
                                         token_type_ids=all_token_type_ids, onthot_ans_type=all_onthot_ans_type)

                start_logits, end_logits = outputs[0], outputs[1]

                start_logits = torch.softmax(start_logits, dim=-1)
                end_logits = torch.softmax(end_logits, dim=-1)
                start_logits = start_logits.argmax(dim=-1)
                end_logits = end_logits.argmax(dim=-1)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                start_logits = to_numpy(start_logits)
                end_logits = to_numpy((end_logits))

                for px, tweet in enumerate(all_orig_tweet):
                    selected_tweet = all_orig_selected[px]
                    _, final_text = calculate_jaccard_score(
                        original_tweet=tweet,
                        target_string=selected_tweet,
                        idx_start=start_logits[px],
                        idx_end=end_logits[px],
                        tokenizer=self.tokenizer,
                    )
                    if (sentiment[px] == "neutral" or len(all_orig_tweet[px].split()) < 3):
                        final_text = tweet
                    all_results.append(final_text)

        # save csv
        submission = pd.read_csv(os.path.join(self.config.data_path, "sample_submission.csv"))

        for i in range(len(submission)):
            final_text = " ".join(set(all_results[i].lower().split()))
            submission.loc[i, 'selected_text'] = final_text

        submission.to_csv(os.path.join(self.config.checkpoint_folder, "submission_{}.csv".format(self.config.fold)))

        return

    def find_errors(self):

        scores = []
        bad_predictions = []
        bad_labels = []
        bad_scores = []

        for fold in range(5):
            checkpoint_folder = os.path.join(self.config.checkpoint_folder_all_fold, 'fold_' + str(fold) + '/')
            val_pred = pd.read_csv(os.path.join(checkpoint_folder, "val_prediction_{}_{}.csv".format(self.config.seed,
                                                                                                   fold)))
            val_label = pd.read_csv(os.path.join(checkpoint_folder, "val_fold_{}_seed_{}.csv".format(fold,
                                                                                                     self.config.seed)))
            pred_text = val_pred.selected_text
            label_text = val_label.selected_text

            for i, label_string in enumerate(label_text):
                pred_string = pred_text[i]

                try:
                    if pred_string[0] == " ":
                        print(pred_string[1:])
                        print(label_string)
                        print(jaccard(label_string.strip(), pred_string.strip()), jaccard(label_string.strip(), pred_string[1:].strip()))
                        print("_______________________________________")
                        pred_string = pred_string[1:]
                    jac = jaccard(label_string.strip(), pred_string.strip())
                except:
                    continue

                scores.append(jac)

                if jac < 0.5:
                    bad_scores.append(jac)
                    bad_predictions.append(pred_string)
                    bad_labels.append(label_string)

        bad_samples = pd.DataFrame({"score": bad_scores, "prediction": bad_predictions, "label": bad_labels})
        bad_samples = bad_samples.sort_values(by=["score"])

        bad_samples.to_csv(os.path.join(self.config.checkpoint_folder_all_fold, "bad_samples.csv"))

        print(np.mean(scores))

        return


if __name__ == "__main__":
    args = parser.parse_args()

    # update fold
    config = Config_Bert(args.fold, model_type=args.model_type, seed=args.seed, batch_size=args.batch_size,
                         accumulation_steps=args.accumulation_steps, Datasampler=args.Datasampler)
    seed_everything(config.seed)
    qa = QA(config)
    qa.train_op()
    # qa.evaluate_op()
    # qa.infer_op()
    # qa.find_errors()
