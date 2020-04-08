from transformers import *
import torch
import torch.nn as nn

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


def jaccard(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Source:
        https://github.com/catalyst-team/catalyst/
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    intersection = torch.sum(gt * pr, dim=-1)
    union = torch.sum(gt, dim=-1) + torch.sum(pr, dim=-1) - intersection + eps
    return (intersection + eps) / union


class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        loss = 1 - jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)
        return loss

############################################ Define Net Class
class TweetBert(nn.Module):
    def __init__(self, model_type="bert-large-uncased", hidden_layers=None):
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

        self.activation = nn.ReLU()

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

        # hidden_states = outputs[0]
        # logits = self.qa_outputs(hidden_states)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

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
            label_mask = torch.zeros_like(start_logits)
            for i in range(label_mask.shape[0]):
                label_mask[i, start_positions[i].data : end_positions[i].data] = 1

            if self.training:
                loss_fct = CrossEntropyLossOHEM(ignore_index=ignored_index, top_k=0.75, reduction="mean")
                loss_jaccard = JaccardLoss()
                start_jaccard_loss = loss_jaccard(start_logits, label_mask)
                end_jaccard_loss = loss_jaccard(end_logits, label_mask)
                if sentiment_weight is None:
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                else:
                    start_loss = loss_fct(start_logits, start_positions, sentiment_weight)
                    end_loss = loss_fct(end_logits, end_positions, sentiment_weight)
                    start_jaccard_loss *= sentiment_weight
                    end_jaccard_loss *= sentiment_weight
                ce_loss = (start_loss + end_loss) / 2
                jaccard_loss = (start_jaccard_loss + end_jaccard_loss) / 2
                total_loss = ce_loss + jaccard_loss.mean() * 0.2
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction="none")
                loss_jaccard = JaccardLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                start_jaccard_loss = loss_jaccard(start_logits, label_mask)
                end_jaccard_loss = loss_jaccard(end_logits, label_mask)
                if sentiment_weight is not None:
                    start_loss *= sentiment_weight
                    end_loss *= sentiment_weight
                    start_jaccard_loss *= sentiment_weight
                    end_jaccard_loss *= sentiment_weight
                ce_loss = (start_loss + end_loss) / 2
                jaccard_loss = (start_jaccard_loss + end_jaccard_loss) / 2
                total_loss = ce_loss.mean() + jaccard_loss.mean() * 0.2
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

    model = TweetBert()

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
    
