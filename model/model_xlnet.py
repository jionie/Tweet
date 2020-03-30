from transformers import *
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
import torch
import torch.nn as nn
import torch.nn.functional as F


############################################ Define Net Class
class TweetXlNet(nn.Module):
    def __init__(self, model_type="xlnet-base-cased", hidden_layers=None):
        super(TweetXlNet, self).__init__()

        self.model_name = 'TweetXlNet'
        self.model_type = model_type

        if hidden_layers is None:
            hidden_layers = [-1, -3, -5, -7]
        self.hidden_layers = hidden_layers

        if model_type == "xlnet-base-cased":
            self.config = XLNetConfig.from_pretrained(model_type)
            self.xlnet = XLNetModel.from_pretrained(model_type, dropout=0, output_hidden_states=True)
        elif model_type == "xlnet-large-cased":
            self.config = XLNetConfig.from_pretrained(model_type)
            self.xlnet = XLNetModel.from_pretrained(model_type, dropout=0, output_hidden_states=True)
        else:
            raise NotImplementedError

        self.start_n_top = self.config.start_n_top
        self.end_n_top = self.config.end_n_top
        self.start_logits = PoolerStartLogits(self.config)
        self.end_logits = PoolerEndLogits(self.config)
        self.answer_class = PoolerAnswerClass(self.config)

        self.down = nn.Linear(len(hidden_layers), 1)
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

    def get_start_logits_by_random_dropout(self, fuse_hidden, p_mask, down, start_logits):

        logit = None
        h = self.activation(down(fuse_hidden)).squeeze(-1)

        for j, dropout in enumerate(self.dropouts):

            if j == 0:
                logit = start_logits(dropout(h), p_mask=p_mask)
            else:
                logit += start_logits(dropout(h), p_mask=p_mask)

        return logit / len(self.dropouts)

    def get_end_logits_by_random_dropout(self, fuse_hidden, start_positions, p_mask, down, end_logits):

        logit = None
        h = self.activation(down(fuse_hidden)).squeeze(-1)

        for j, dropout in enumerate(self.dropouts):

            if j == 0:
                logit = end_logits(dropout(h), start_positions=start_positions, p_mask=p_mask)
            else:
                logit += end_logits(dropout(h), start_positions=start_positions, p_mask=p_mask)

        return logit / len(self.dropouts)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            is_impossible=None,
            cls_index=None,
            p_mask=None,
    ):

        transformer_outputs = self.xlnet(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states_all = transformer_outputs[1]
        hidden_states = self.get_hidden_states(hidden_states_all)

        start_logits = self.get_start_logits_by_random_dropout(hidden_states, p_mask, self.down, self.start_logits)

        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.get_end_logits_by_random_dropout(hidden_states, start_positions, p_mask, self.down,
                                                               self.end_logits)

            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            outputs = (total_loss,) + outputs

        else:
            hidden_states = self.down(hidden_states).squeeze(-1)
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum(
                "blh,bl->bh", hidden_states, start_log_probs
            )  # get the representation of START as weighted sum of hidden states
            cls_logits = self.answer_class(
                hidden_states, start_states=start_states, cls_index=cls_index
            )  # Shape (batch size,): one single `cls_logits` for each sample

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs


############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    all_input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]])
    all_attention_masks = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]])
    all_token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    all_start_positions = torch.tensor([0, 1])
    all_end_positions =  torch.tensor([3, 4])
    print(all_start_positions.shape)

    model = TweetXlNet()

    print("------------------------testing loss----------------------")
    y = model(input_ids=all_input_ids, attention_mask=all_attention_masks, token_type_ids=all_token_type_ids,
              start_positions=all_start_positions, end_positions=all_end_positions)
    print(len(y))
    print("loss: ", y[0])
    print("------------------------testing finished----------------------")

    print("------------------------testing inference----------------------")
    y = model(input_ids=all_input_ids, attention_mask=all_attention_masks, token_type_ids=all_token_type_ids)
    print(len(y))
    print("start index: ", y[1])
    print("end index: ", y[3])
    print("------------------------testing inference----------------------")

    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net----------------------")
    test_Net()

