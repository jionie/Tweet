from transformers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import *


##################################################### CrossEntropyLossOHEM
class CrossEntropyLossOHEM(torch.nn.Module):
    def __init__(self, ignore_index, top_k=0.75, reduction='mean'):
        super(CrossEntropyLossOHEM, self).__init__()
        self.top_k = top_k
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.reduction = reduction

    def forward(self, input, target, sentiment=None, position_penalty=None):
        loss = self.loss(input, target)
        if sentiment is not None:
            loss *= sentiment
        if position_penalty is not None:
            loss *= position_penalty
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


def pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
    # neg_weight for when pred position < target position
    # pos_weight for when pred position > target position
    gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
    gap = gap.type(torch.float32)
    return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap)


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

        self.qa_start = nn.Linear(self.config.hidden_size, 1)
        self.qa_end = nn.Linear(self.config.hidden_size, 1)
        self.qa_classifier = nn.Linear(self.config.hidden_size, 3)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

        self.backup = {}

        self.aoa = AttentionOverAttention(self.config.hidden_size)

        self.cross_attention = CrossAttention(num_attention_head=12,
                                              hidden_dim=self.config.hidden_size,
                                              dropout_rate=0.1)

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.bert.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if not torch.isnan(norm):
                    r_at = epsilon * param.grad / (norm + 1e-8)
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.bert.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = self.backup[name]
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
            onthot_ans_type=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            sentiment_weight=None,
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

        # # hidden for context, padding added, ignore end_idx
        fuse_hidden_context = fuse_hidden[:, 4:-1, :]

        # #################################################################### aoa, attention over attention
        # hidden for question, cls + sentiment
        # fuse_hidden_question = fuse_hidden[:, 0:2, :]
        # aoa_s_start, aoa_s_end = self.aoa(fuse_hidden_question, fuse_hidden_context, attention_mask[:, 4:-1])
        # start_logits = self.get_logits_by_random_dropout(fuse_hidden_context, self.qa_start).squeeze(-1) * aoa_s_start
        # end_logits = self.get_logits_by_random_dropout(fuse_hidden_context, self.qa_end).squeeze(-1) * aoa_s_end

        # #################################################################### cross attention
        # hidden for question, cls + sentiment
        # fuse_hidden_question = fuse_hidden[:, 0:2, :]
        # fuse_hidden_context_dot = self.cross_attention(fuse_hidden_context, fuse_hidden_question, fuse_hidden_question,
        #                                                attention_mask[:, 4:-1])
        # start_logits = self.get_logits_by_random_dropout(fuse_hidden_context_dot, self.qa_start).squeeze(-1)
        # end_logits = self.get_logits_by_random_dropout(fuse_hidden_context_dot, self.qa_end).squeeze(-1)

        # #################################################################### direct approach
        start_logits = self.get_logits_by_random_dropout(fuse_hidden_context, self.qa_start).squeeze(-1)
        end_logits = self.get_logits_by_random_dropout(fuse_hidden_context, self.qa_end).squeeze(-1)

        classification_logits = self.get_logits_by_random_dropout(fuse_hidden[:, 0, :], self.qa_classifier)

        outputs = (start_logits, end_logits, classification_logits) + outputs[2:]
        # outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            start_position_penalty = pos_weight(start_logits, start_positions, 1, 1)
            end_position_penalty = pos_weight(end_logits, end_positions, 1, 1)

            if self.training:
                loss_fct = CrossEntropyLossOHEM(ignore_index=ignored_index, top_k=0.75, reduction="mean")

                loss_classification = nn.BCEWithLogitsLoss()
                classification_loss = loss_classification(classification_logits, onthot_ans_type)

                if sentiment_weight is None:
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                else:
                    start_loss = loss_fct(start_logits, start_positions, sentiment_weight)
                    end_loss = loss_fct(end_logits, end_positions, sentiment_weight)
                total_loss = (start_loss + end_loss + classification_loss) / 3
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction="none")
                loss_classification = nn.BCEWithLogitsLoss()

                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                classification_loss = loss_classification(classification_logits, onthot_ans_type)

                if sentiment_weight is not None:
                    start_loss *= sentiment_weight
                    end_loss *= sentiment_weight

                total_loss = (start_loss.mean() + end_loss.mean() + classification_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
        
        

############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    all_input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]])
    all_attention_masks = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]])
    all_token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    all_start_positions = torch.tensor([0, 1])
    all_end_positions =  torch.tensor([2, 3])
    all_onthot_ans_type = torch.tensor([[0, 0, 1], [1, 0, 0]]).float()
    print(all_start_positions.shape)

    model = TweetBert(max_seq_len=7)

    y = model(input_ids=all_input_ids, attention_mask=all_attention_masks, token_type_ids=all_token_type_ids,
              start_positions=all_start_positions, end_positions=all_end_positions, onthot_ans_type=all_onthot_ans_type)

    print("loss: ", y[0])
    print("start_logits: ", y[1])
    print("end_logits: ", y[2])
    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net----------------------")
    test_Net()
    
