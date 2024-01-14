from torch import nn
from transformers.models.distilbert import DistilBertPreTrainedModel, DistilBertConfig, DistilBertModel
from transformers import DataCollatorWithPadding
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput
from torch.nn import BCEWithLogitsLoss
from dataclasses import dataclass
from torch.autograd import Variable
import torch


def to_list(tensor_or_iterable):
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.tolist()
    return list(tensor_or_iterable)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


@dataclass
class GazetteerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GazetteerConfig(DistilBertConfig):
    def __init__(self, num_labels_intents=None, index2intent=None, intent2index=None, use_gaz_embeds=True, gaz_embeds_dim=None,
                 index2tag=None, tag2index=None, num_labels_tags=None, batch_size=None, use_gaz_features=True, **kwargs):
        super().__init__(**kwargs)
        self.num_labels_intents = num_labels_intents
        self.index2intent = index2intent
        self.intent2index = intent2index
        self.num_labels_tags = num_labels_tags
        self.index2tag = index2tag
        self.tag2index = tag2index
        self.batch_size = batch_size
        self.use_gaz_features = use_gaz_features
        self.use_gaz_embeds = use_gaz_embeds
        self.gaz_embeds_dim = gaz_embeds_dim
        

class DistilBertForGazetteer(DistilBertPreTrainedModel):
    def __init__(self, config: GazetteerConfig):
        super().__init__(config)
        self.num_labels_intents = config.num_labels_intents
        self.index2intent = config.index2intent
        self.intent2index = config.intent2index
        self.num_labels_tags = config.num_labels_tags
        self.index2tag = config.index2tag
        self.tag2index = config.tag2index
        self.config = config

        self.distilbert = DistilBertModel(config)
        
        if config.use_gaz_features:
            if config.use_gaz_embeds:
                self.embeddings = nn.Embedding(self.num_labels_tags + 1, config.gaz_embeds_dim)
                lstm_input_dim = config.dim + config.gaz_embeds_dim
            else:
                self.onehot2lstm = nn.Linear(config.dim + self.num_labels_tags, config.dim)
                lstm_input_dim = config.dim
        else:
            lstm_input_dim = config.dim
        
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=config.dim, bidirectional=True)
        self.hidden2label = nn.Linear(config.dim * 2, config.dim)

        self.classifier = nn.Linear(config.dim, config.num_labels_intents)        
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        
        self.hidden = [Variable(torch.zeros(2, config.batch_size, config.dim).cuda()),
                    Variable(torch.zeros(2, config.batch_size, config.dim).cuda())]

        self.post_init()
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                    if 'weight_hh' in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                    if 'bias' in name:
                        param.data.fill_(0)

    def get_position_embeddings(self) -> nn.Embedding:
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        gaz_features,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[GazetteerOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]
        hidden_state = hidden_state.view(hidden_state.size(1), hidden_state.size(0), -1)
        
        if self.config.use_gaz_features:
            if self.config.use_gaz_embeds:
                gaz_features = self.embeddings(gaz_features).view(hidden_state.size(0), hidden_state.size(1), -1)
                lstm_input = torch.cat((hidden_state, gaz_features), 2)
            else:
                lstm_input = torch.cat((hidden_state, gaz_features), 2)
                lstm_input = self.onehot2lstm(lstm_input)
        else:
            lstm_input = hidden_state
            
        self.hidden = repackage_hidden(self.hidden)
        lstm_out, hidden = self.lstm(lstm_input, (self.hidden[0][:,:hidden_state.size(1)], self.hidden[1][:,:hidden_state.size(1)]))
        self.hidden[0][:,:hidden_state.size(1)], self.hidden[1][:,:hidden_state.size(1)] = hidden
        classification_output = self.hidden2label(lstm_out[-1])
        classification_output = nn.ReLU()(classification_output)
        classification_output = self.dropout(classification_output)
        classification_logits = self.classifier(classification_output)

        loss = None
        if label is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(classification_logits, label)

        if not return_dict:
            output = (classification_logits, ) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return GazetteerOutput(
            loss=loss,
            logits=classification_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class GazetteerDataCollator(DataCollatorWithPadding):
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"
    
    def __init__(self, num_labels_tags, is_embeds=True, **kwargs):
        super().__init__(**kwargs)
        self.num_labels_tags = num_labels_tags
        self.is_embeds = is_embeds
        
    def __call__(self, features):
        tagger_labels = [feature["gaz_features"] for feature in features] if "gaz_features" in features[0].keys() else None
        no_labels_features = [{k: v for k, v in feature.items() if k != "gaz_features"} for feature in features]
        
        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        if padding_side == "right":
            batch["gaz_features"] = [to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in tagger_labels]
        else:
            batch["gaz_features"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in tagger_labels]

        batch["gaz_features"] = torch.tensor(batch["gaz_features"], dtype=torch.int64)
        if self.is_embeds:
            batch["gaz_features"][batch["gaz_features"] == self.label_pad_token_id] = self.num_labels_tags
        else:
            gaz_onehot = torch.zeros((batch["gaz_features"].size(1), batch["gaz_features"].size(0), self.num_labels_tags), dtype=torch.int64)
            for i, elem in enumerate(batch["gaz_features"]):
                for j, feature in enumerate(elem):
                    if feature != self.label_pad_token_id:
                        gaz_onehot[j, i, feature] = 1
            batch["gaz_features"] = gaz_onehot
        
        return batch
