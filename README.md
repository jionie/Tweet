# Tweet Sentiment Extraction
This repository contains codes for Tweet-Sentiment-Extraction Competition. 

## Structure for data
please arrange project folder as 
```plain
codes
└── all codes in this repo
```
```plain
input
└── tweet-sentiment-extraction
      ├── train.csv
      ├── test.csv
      ├── train_clean_v03.csv 
      ├── pseudo_labels.csv
      ├── sample_submission.csv
      └── split
           └── ...
```
```plain
model
└── TweetBert
    ├── roberta-base-42
    ├── albert-large-1996
    └── ...
```

## Codes for Dataset
Please check codes for Dataset in "dataset" folder, (the unit test function maybe out of date):
```bash
python3 dataset_v2.py
```

## Codes for Model
Please check codes for Model in "model" folder:
```bash
python3 model_bert.py
```

## Codes for Training
Please check codes for Training, you should change the path first then run:
```bash
./k-fold-v2.sh
```
| single model           | hidden_layers | LR (head, backbone) | config.hidden_dropout_prob |
| ---------------- |  ---- | ---- | ---- |
|roberta-base|[-1, -2, -3, -4]|2e-4 and 1e-5|0.1
|albert-large|[-1, -2, -3, -4]|2e-4 and 1e-5|0.1
|xlnet-base|[-1, -2, -3, -4]|2e-4 and 1e-5|0.1


## Pseudo labeling
Codes already include pseudo labeled data from original dataset, you could remove it by changing dataset_v2.py


#### model performace (oof)
| single model           | oof |
| ---------------- |  ---- |
|roberta-base, seed 42|0.722|
|roberta-base, seed 42, 2 rounds pseudo labeling|0.724|
|roberta-base, seed 666|0.724|
|roberta-base, seed 666, 2 rounds pseudo labeling|0.724|
|roberta-base, seed 1234|0.722|
|roberta-base, seed 1234, 2 rounds pseudo labeling|0.722|
|albert-large, seed 1996|0.720|
|albert-large, seed 1996, 2 rounds pseudo labeling|0.722|
|xlnet-base, seed 1997|0.714|


## Codes for inference
Please use "preprocessing-new-pipeline-pseudo-model-ensemble.ipynb", this is available on https://www.kaggle.com/jionie/preprocessing-new-pipeline-pseudo-model-ensemble

## License
[MIT](https://choosealicense.com/licenses/mit/)


