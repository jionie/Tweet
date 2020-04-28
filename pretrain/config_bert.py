import os


class Config_Bert:
    # config settings
    def __init__(self, fold, model_type="roberta-base", seed=2020, batch_size=16, accumulation_steps=1):
        # setting
        self.reuse_model = True
        self.load_from_load_from_data_parallel = False
        self.data_parallel = False  # enable data parallel training
        self.apex = True  # enable mix precision training
        self.adversarial = False  # enable adversarial training
        self.load_optimizer = False
        self.skip_layers = []
        # model
        self.model_type = model_type
        self.do_lower_case = True
        self.model_name = 'TweetBert'
        self.hidden_layers = [-1, -2, -3, -4]
        self.sentiment_weight_map = {"neutral": 0.8, "positive": 1, "negative": 1}
        # path, specify the path for data
        self.data_path = '/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/'
        # path, specify the path for saving splitted csv
        self.save_path = '/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/'
        # k fold setting
        self.split = "StratifiedKFold"
        self.seed = seed
        self.n_splits = 5
        self.fold = fold
        # path, specify the path for saving model
        self.model_folder = os.path.join("/media/jionie/my_disk/Kaggle/Tweet/pretrain/model", self.model_name)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        self.checkpoint_folder_all_fold = os.path.join(self.model_folder, self.model_type + '-' + str(self.seed))
        if not os.path.exists(self.checkpoint_folder_all_fold):
            os.mkdir(self.checkpoint_folder_all_fold)
        self.checkpoint_folder = os.path.join(self.checkpoint_folder_all_fold,'fold_' + str(self.fold) + '/')
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        self.save_point = os.path.join(self.checkpoint_folder, '{}_step_{}_epoch.pth')
        self.load_points = [p for p in os.listdir(self.checkpoint_folder) if p.endswith('.pth')]
        if len(self.load_points) != 0:
            self.load_point = sorted(self.load_points, key=lambda x: int(x.split('_')[0]))[-1]
            self.load_point = os.path.join(self.checkpoint_folder, self.load_point)
        else:
            self.reuse_model = False
        # dataset setting
        self.max_seq_length = 192
        self.max_query_length = 64
        self.max_answer_length = 30
        self.doc_stride = 128
        self.threads = 4
        # optimizer
        self.optimizer_name = "AdamW"
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 2
        # lr scheduler, can choose to use proportion or steps
        self.lr_scheduler_name = 'WarmupLinear'
        self.warmup_proportion = 0.005
        self.warmup_steps = 200
        # lr
        self.lr = 2e-4
        self.weight_decay = 0.001
        # differential lr settings
        self.differential_lr = False
        self.decay_factor = 0.9
        self.min_lr = 1e-5
        # differential lr setting, step or decay
        self.method = "step"
        # dataloader settings
        self.batch_size = batch_size
        self.val_batch_size = 32
        self.num_workers = 4
        self.shuffle = True
        self.drop_last = True
        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        # epochs
        self.num_epoch = 8
        # saving rate
        self.saving_rate = 1 / 3
        # early stopping
        self.early_stopping = 3 / self.saving_rate
        # progress rate
        self.progress_rate = 1/10
        # inference setting
        self.n_best_size = 10
        self.verbose_logging = False
        self.null_score_diff_threshold = 0
        self.version_2_with_negative = False
