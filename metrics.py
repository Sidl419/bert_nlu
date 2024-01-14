from sklearn.metrics import roc_auc_score
import numpy as np
import evaluate
import json
import torch


seqeval = evaluate.load('seqeval')
with open("./data/ontology.json") as f:
    ontology = json.load(f)
slots = ["O"]
for slot_name in ontology['slots'].keys():
    slots.append("B-" + slot_name)
    slots.append("I-" + slot_name)
label_names = slots


def multi_label_metrics(eval_pred):
    predictions, labels = eval_pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    roc_auc = roc_auc_score(labels, probs, average = 'micro')

    return {"classification_roc_auc": roc_auc}


def ner_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    true_labels = [[label_names[l] for l in label if l!=-100] for label in labels]
    true_predictions = [[label_names[p] for p,l in zip(prediction, label) if l!=-100] 
                        for prediction, label in zip(predictions, labels)]

    all_metrics = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {"tagger_precision": all_metrics['overall_precision'],
            "tagger_recall": all_metrics['overall_recall'],
            "tagger_f1": all_metrics['overall_f1'],
            "tagger_accuracy": all_metrics['overall_accuracy']}


def multitask_metrics(eval_pred):
    predictions_1, predictions_2 = eval_pred[0]
    labels_1, labels_2 = eval_pred[1]

    classification_metrics = multi_label_metrics((predictions_1, labels_1))
    tagger_metrics = ner_metrics((predictions_2, labels_2))

    return {**classification_metrics, **tagger_metrics}
