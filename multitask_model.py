from torch import nn
from transformers.models.distilbert import DistilBertPreTrainedModel, DistilBertConfig, DistilBertModel
from transformers import DataCollatorWithPadding
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from dataclasses import dataclass
import torch


def to_list(tensor_or_iterable):
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.tolist()
    return list(tensor_or_iterable)

@dataclass
class MultiTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_1: torch.FloatTensor = None
    logits_2: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MultiTaskConfig(DistilBertConfig):
    def __init__(self, num_labels_intents=None, index2intent=None, intent2index=None, 
                 index2tag=None, tag2index=None, num_labels_tags=None, **kwargs):
        super().__init__(**kwargs)
        self.num_labels_intents = num_labels_intents
        self.index2intent = index2intent
        self.intent2index = intent2index
        self.num_labels_tags = num_labels_tags
        self.index2tag = index2tag
        self.tag2index = tag2index


class DistilBertForMultiTask(DistilBertPreTrainedModel):
    def __init__(self, config: MultiTaskConfig):
        super().__init__(config)
        self.num_labels_intents = config.num_labels_intents
        self.index2intent = config.index2intent
        self.intent2index = config.intent2index
        self.num_labels_tags = config.num_labels_tags
        self.index2tag = config.index2tag
        self.tag2index = config.tag2index
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels_intents)
        self.tagger = nn.Linear(config.dim, config.num_labels_tags)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        label_1: Optional[torch.LongTensor] = None,
        label_2: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MultiTaskOutput, Tuple[torch.Tensor, ...]]:
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

        classification_output = self.pre_classifier(hidden_state[:, 0])
        classification_output = nn.ReLU()(classification_output)
        classification_output = self.dropout(classification_output)
        classification_logits = self.classifier(classification_output)

        tagging_output = self.dropout(hidden_state)
        tagging_logits = self.tagger(tagging_output)

        loss = None
        if label_1 is not None and label_2 is not None:
            classification_loss_fct = BCEWithLogitsLoss()
            classification_loss = classification_loss_fct(classification_logits, label_1)
            
            tagging_loss_fct = CrossEntropyLoss()
            tagging_loss = tagging_loss_fct(tagging_logits.view(-1, self.num_labels_tags), label_2.view(-1))

            loss = classification_loss + tagging_loss

        if not return_dict:
            output = (classification_logits, tagging_logits) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return MultiTaskOutput(
            loss=loss,
            logits_1=classification_logits,
            logits_2=tagging_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class MultiTaskDataCollator(DataCollatorWithPadding):
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"
        
    def __call__(self, features):
        tagger_labels = [feature["label_2"] for feature in features] if "label_2" in features[0].keys() else None
        no_labels_features = [{k: v for k, v in feature.items() if k != "label_2"} for feature in features]
        
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
            batch["label_2"] = [to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in tagger_labels]
        else:
            batch["label_2"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in tagger_labels]

        batch["label_2"] = torch.tensor(batch["label_2"], dtype=torch.int64)
        return batch
