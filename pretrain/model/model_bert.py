from transformers import *
import math
import torch
import torch.nn as nn


##################################################### CrossEntropyLossOHEM
class CrossEntropyLossOHEM(torch.nn.Module):
    def __init__(self, ignore_index, top_k=0.75, reduction='mean'):
        super(CrossEntropyLossOHEM, self).__init__()
        self.top_k = top_k
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.reduction = reduction

    def forward(self, input, target, sentiment=None):
        loss = self.loss(input, target)
        if sentiment is not None:
            loss *= sentiment
        if self.top_k == 1:
            if self.reduction == "mean":
                return torch.mean(loss)
            elif self.reduction == "sum":
                return torch.sum(loss)
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]), dim=0)
            if self.reduction == "mean":
                return torch.mean(valid_loss)
            elif self.reduction == "sum":
                return torch.sum(valid_loss)
            elif self.reduction == "none":
                return valid_loss
            else:
                raise NotImplementedError

def swish(x):
    return x * torch.sigmoid(x)


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


############################################ Define Net Class
class TweetBert(nn.Module):
    def __init__(self, model_type="bert-large-uncased", max_seq_len=192, hidden_layers=None):
        super(TweetBert, self).__init__()

        self.model_name = 'TweetBert'
        self.model_type = model_type
        self.max_seq_len = max_seq_len

        if hidden_layers is None:
            hidden_layers = [-1, -2, -3]
        self.hidden_layers = hidden_layers

        if model_type == "bert-large-uncased":
            self.config = BertConfig.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
            self.bert = BertModel.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad",
                                                  hidden_dropout_prob=0.2, output_hidden_states=True)
        elif model_type == "bert-large-cased":
            self.config = BertConfig.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
            self.bert = BertModel.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad",
                                                        hidden_dropout_prob=0.2, output_hidden_states=True)
        elif model_type == "bert-base-uncased":
            self.config = AutoConfig.from_pretrained(
                "bert-base-uncased",
            )
            self.config.hidden_dropout_prob = 0.1
            self.config.output_hidden_states = True
            self.bert = AutoModel.from_pretrained(
                model_type,
                config=self.config,
            )
        elif model_type == "bert-base-cased":
            self.config = AutoConfig.from_pretrained(
                "bert-base-cased",
            )
            self.config.hidden_dropout_prob = 0.1
            self.config.output_hidden_states = True
            self.bert = AutoModel.from_pretrained(
                model_type,
                config=self.config,
            )
        elif model_type == "roberta-base":
            self.config = AutoConfig.from_pretrained(
                "roberta-base",
            )
            self.config.hidden_dropout_prob = 0.1
            self.config.output_hidden_states = True
            self.bert = AutoModel.from_pretrained(
                model_type,
                config=self.config,
            )
        elif model_type == "roberta-base-squad":
            self.config = AutoConfig.from_pretrained(
                "deepset/roberta-base-squad2",
            )
            self.config.hidden_dropout_prob = 0.1
            self.config.output_hidden_states = True
            self.bert = AutoModel.from_pretrained(
                "deepset/roberta-base-squad2",
                config=self.config,
            )
        elif model_type == "roberta-large":
            self.config = AutoConfig.from_pretrained(
                "roberta-large",
            )
            self.config.hidden_dropout_prob = 0.1
            self.config.output_hidden_states = True
            self.bert = AutoModel.from_pretrained(
                model_type,
                config=self.config,
            )
        else:
            raise NotImplementedError

        # hidden states fusion
        weights_init = torch.zeros(len(hidden_layers)).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.qa_classifier = nn.Linear(self.config.hidden_size, 13)

        def init_weights(m):
            if type(m) == nn.Linear:
                # torch.nn.init.xavier_uniform(m.weight)
                # m.bias.data.fill_(0)
                torch.nn.init.normal_(m.weight, std=0.02)

        self.qa_classifier.apply(init_weights)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

        self.backup = {}

    def get_hidden_states(self, hidden_states):

        fuse_hidden = None
        # concat hidden
        for i in range(len(self.hidden_layers)):
            if i == 0:
                hidden_layer = self.hidden_layers[i]
                fuse_hidden = hidden_states[hidden_layer].unsqueeze(-1)
            else:
                hidden_layer = self.hidden_layers[i]
                hidden_state = hidden_states[hidden_layer].unsqueeze(-1)
                fuse_hidden = torch.cat([fuse_hidden, hidden_state], dim=-1)

        fuse_hidden = (torch.softmax(self.layer_weights, dim=0) * fuse_hidden).sum(-1)

        return fuse_hidden

    def get_logits_by_random_dropout(self, fuse_hidden, fc):

        logit = None
        h = fuse_hidden

        for j, dropout in enumerate(self.dropouts):

            if j == 0:
                logit = fc(dropout(h))
            else:
                logit += fc(dropout(h))

        return logit / len(self.dropouts)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            onthot_sentiments_type=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[2]
        # bs, seq len, hidden size
        fuse_hidden = self.get_hidden_states(hidden_states)
        # fuse_hidden = fuse_hidden.mean(1)

        classification_logits = self.get_logits_by_random_dropout(fuse_hidden[:, 0, :], self.qa_classifier)

        loss_classification = nn.BCEWithLogitsLoss()
        classification_loss = loss_classification(classification_logits, onthot_sentiments_type)

        outputs = (classification_logits, ) + outputs[2:]
        outputs = (classification_loss,) + outputs

        return outputs
