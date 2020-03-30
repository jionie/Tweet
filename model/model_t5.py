from transformers import *
import torch
import torch.nn as nn


############################################ Define Net Class
class TweetT5(nn.Module):
    def __init__(self, model_type="t5-base", hidden_layers=None):
        super(TweetT5, self).__init__()

        self.model_name = 'TweetT5'
        self.model_type = model_type

        if hidden_layers is None:
            hidden_layers = [-1, -3, -5, -7]
        self.hidden_layers = hidden_layers

        if model_type == "t5-large":
            self.config = T5Config.from_pretrained(model_type)
            self.t5 = T5Model.from_pretrained(model_type, dropout_rate=0.1, output_hidden_states=True)
        elif model_type == "t5-base":
            self.config = T5Config.from_pretrained(model_type)
            self.t5 = T5Model.from_pretrained(model_type, dropout_rate=0.1, output_hidden_states=True)
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
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            start_positions=None,
            end_positions=None,
    ):

        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
        )

        hidden_states = outputs[2]

        fuse_hidden = self.get_hidden_states(hidden_states)
        logits = self.get_logits_by_random_dropout(fuse_hidden, self.down, self.qa_outputs)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    all_input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]])
    all_attention_masks = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]])
    all_token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    all_start_positions = torch.tensor([0, 1])
    all_end_positions = torch.tensor([3, 4])
    print(all_start_positions.shape)

    model = TweetT5()

    y = model(input_ids=all_input_ids, attention_mask=all_attention_masks, start_positions=all_start_positions,
              end_positions=all_end_positions)
    print(y)
    print("loss: ", y[0])
    print("start_logits: ", y[1])
    print("end_logits: ", y[2])
    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net----------------------")
    test_Net()

