from transformers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


##################################################### HeatMapLoss -> MSELoss or KLDivLoss
def get_norm_prob(x, mu, sigma=0.15):
    left = 1 / (np.sqrt(2 * np.pi)) * np.sqrt(sigma)
    right = np.exp(-(x - mu)**2) / (2 * sigma)
    return left * right


def get_smooth_gt(position, seq_len=5, sigma=0.5):
    norm = [get_norm_prob(i, position, sigma) for i in range(seq_len)]
    norm /= sum(norm)
    norm = torch.tensor(norm, dtype=torch.float)
    return norm


class KLDivLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, model_output, target):
        log = F.log_softmax(model_output, dim=1)
        loss = self.loss(log, target)
        return loss


##################################################### LovaszLoss
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors


def lovasz_hinge_flat(logits, labels, ignore_index):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    errors = hinge(logits, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class LovaszLoss(nn.Module):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    __name__ = 'LovaszLoss'

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        return lovasz_hinge_flat(logits, labels, self.ignore_index)

############################################ Define Net Class
class TweetBert(nn.Module):
    def __init__(self, model_type="bert-large-uncased", max_seq_len=192, hidden_layers=None):
        super(TweetBert, self).__init__()

        self.model_name = 'TweetBert'
        self.model_type = model_type

        if hidden_layers is None:
            hidden_layers = [-1]
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

        self.down = nn.Linear(len(hidden_layers), 1)
        self.qa_outputs = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.qa_segment = nn.Linear(2 * max_seq_len, max_seq_len)

        self.activation = nn.ReLU()
        self.activation_seg = nn.SELU()

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

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

        return fuse_hidden

    def get_logits_by_random_dropout(self, fuse_hidden, down, fc):

        logit = None
        h = self.activation(down(fuse_hidden)).squeeze(-1)

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

        fuse_hidden = self.get_hidden_states(hidden_states)
        logits = self.get_logits_by_random_dropout(fuse_hidden, self.down, self.qa_outputs)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        seq_len = start_logits.shape[1]

        segment_logits = self.qa_segment(self.activation_seg(torch.cat((start_logits, end_logits), dim=1)))

        outputs = (start_logits, end_logits,) + outputs[2:]
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
            smoothed_start_position = torch.zeros_like(start_logits)
            smoothed_end_position = torch.zeros_like(end_logits)
            # for segmentation
            label_mask = torch.zeros_like(start_logits)
            for i in range(label_mask.shape[0]):
                label_mask[i, start_positions[i].data: end_positions[i].data] = 1

            for i in range(start_logits.shape[0]):
                smoothed_start_position[i] = get_smooth_gt(start_positions[i].item(), seq_len, 0.25)
                smoothed_end_position[i] = get_smooth_gt(end_positions[i].item(), seq_len, 0.25)
            smoothed_start_position = smoothed_start_position
            smoothed_end_position = smoothed_end_position

            if self.training:
                loss_fct = CrossEntropyLossOHEM(ignore_index=ignored_index, top_k=0.75, reduction="mean")
                loss_heatmap = KLDivLoss(reduction="none")
                loss_lovasz = LovaszLoss(ignore_index=ignored_index)
                smoothed_start_loss = loss_heatmap(start_logits, smoothed_start_position).mean(dim=1)
                smoothed_end_loss = loss_heatmap(end_logits, smoothed_end_position).mean(dim=1)
                segment_loss = loss_lovasz(segment_logits, label_mask)
                if sentiment_weight is None:
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                else:
                    start_loss = loss_fct(start_logits, start_positions, sentiment_weight)
                    end_loss = loss_fct(end_logits, end_positions, sentiment_weight)
                    smoothed_start_loss *= sentiment_weight
                    smoothed_end_loss *= sentiment_weight
                ce_loss = (start_loss + end_loss) / 2
                heapmap_loss = (smoothed_start_loss + smoothed_end_loss) / 2
                total_loss = 0.7 * ce_loss + 0.3 * heapmap_loss.mean() + 0.1 * segment_loss
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction="none")
                loss_heatmap = KLDivLoss(reduction="none")
                loss_lovasz = LovaszLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                smoothed_start_loss = loss_heatmap(start_logits, smoothed_start_position).mean(dim=1)
                smoothed_end_loss = loss_heatmap(end_logits, smoothed_end_position).mean(dim=1)
                segment_loss = loss_lovasz(segment_logits, label_mask)
                if sentiment_weight is not None:
                    start_loss *= sentiment_weight
                    end_loss *= sentiment_weight
                    smoothed_start_loss *= sentiment_weight
                    smoothed_end_loss *= sentiment_weight
                ce_loss = (start_loss + end_loss) / 2
                heapmap_loss = (smoothed_start_loss + smoothed_end_loss) / 2
                total_loss = 0.7 * ce_loss.mean() + 0.3 * heapmap_loss.mean() + 0.1 * segment_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
        
        

############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    all_input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]])
    all_attention_masks = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]])
    all_token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    all_start_positions = torch.tensor([0, 1])
    all_end_positions =  torch.tensor([3, 4])
    print(all_start_positions.shape)

    model = TweetBert(max_seq_len=7)

    y = model(input_ids=all_input_ids, attention_mask=all_attention_masks, token_type_ids=all_token_type_ids,
              start_positions=all_start_positions, end_positions=all_end_positions)

    print("loss: ", y[0])
    print("start_logits: ", y[1])
    print("end_logits: ", y[2])
    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net----------------------")
    test_Net()
    
