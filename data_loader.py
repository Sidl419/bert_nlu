from typing import Optional, Callable
from datasets import Dataset
import json
import os


_LOW, _MID, _LARGE = "low", "mid", "large"  # Data regimes
_BANKING, _HOTELS, _ALL, _HOTELS_BANKING, _BANKING_HOTELS = (
"banking", "hotels", "all", "hotels-banking", "banking-hotels")  # Domains


class DataLoader:
    def __init__(self, data_dir: str, tokenizer: Callable, tokenizer_params: dict) -> None:
        """Modified loader for the NLU++ data

        Args:
            data_dir: directory with the NLU++ data 
            tokenizer: tokenizer for your model (used for right tagging labeling)
        """
        self._data = self._read_data(data_dir)
        with open(os.path.join(data_dir, f"ontology.json")) as f:
            self.ontology = json.load(f)

        self.intents = ['other'] + list(self.ontology['intents'].keys())

        self.index2intent = {idx:tag for idx, tag in enumerate(self.intents)}
        self.intent2index = {tag:idx for idx, tag in enumerate(self.intents)}

        self.slots = ["O"]
        for slot_name in self.ontology['slots'].keys():
            self.slots.append("B-" + slot_name)
            self.slots.append("I-" + slot_name)

        self.index2tag = {idx:tag for idx, tag in enumerate(self.slots)}
        self.tag2index = {tag:idx for idx, tag in enumerate(self.slots)}

        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params

    @staticmethod
    def _read_data(data_dir: str) -> dict:
        data = {}

        for domain in [_BANKING, _HOTELS]:
            data[domain] = {}
            for fold in range(20):
                with open(os.path.join(
                        data_dir, domain, f"fold{fold}.json")) as f:
                    data[domain][fold] = json.load(f)

        return data
    
    @staticmethod
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word=None

        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)

            elif word_id is None:
                new_labels.append(-100)

            else:
                label = labels[word_id]
                if label%2==1:
                    label = label + 1
                new_labels.append(label)

        return new_labels
    
    def _process_sample(self, input: dict) -> dict:
        sample = input.copy()
        if isinstance(sample['text'], list):
            sample['text'] = ' '.join(sample['text'])
        res = ['O'] * len(sample['text'].split())

        if 'slots' not in sample:
            sample['slots'] = dict()

        for name, value in sample['slots'].items():
            first_token = len(sample['text'][:value['span'][0]].split())
            last_token = len(sample['text'][:value['span'][1] + 1].split())

            res[first_token] = "B-" + name
            for token in range(first_token + 1, last_token):
                res[token] = "I-" + name

        del sample['slots']

        sample['tagger'] = res
        sample['tokens'] = ['[CLS]'] + sample['text'].split() + ['[SEP]']

        if 'intents' not in sample:
            sample['intents'] = ['other']
        
        sample['classification_labels'] = [0.0] * len(self.index2intent)
        for intent in sample['intents']:
            sample['classification_labels'][self.intent2index[intent]] = 1.0

        tokens = self.tokenizer(sample['text'].split(), is_split_into_words=True, **self.tokenizer_params)
        sample['input_ids'] = tokens['input_ids']
        sample['attention_mask'] = tokens['attention_mask']
        sample['tagging_labels'] = self.align_labels_with_tokens([self.tag2index[tag] for tag in res], tokens.word_ids())

        return sample

    def _get_cross_domain_data(self, source_domain: str, target_domain: str) -> dict:
        train_examples, test_examples = [], []

        for fold_i in range(20):
            train_examples += self._data[source_domain][fold_i]
            test_examples += self._data[target_domain][fold_i]

        # delete non-generic slots and values
        generic_intents = []
        generic_slots = []

        for intent, metadata in self.ontology["intents"].items():
            if "general" in metadata["domain"]:
                generic_intents.append(intent)

        for slot, metadata in self.ontology["slots"].items():
            if "general" in metadata["domain"]:
                generic_slots.append(slot)

        for example in train_examples + test_examples:
            if "intents" in example:
                example["intents"] = [intent for intent in example["intents"] if intent in generic_intents]
            if "slots" in example:
                example["slots"] = {slot: data for slot, data in example["slots"].items() if slot in generic_slots}

        # keeping the same structure as other experiments, even if there is only 1 fold
        experiment_data = {0: {"train": Dataset.from_list(train_examples), "test": Dataset.from_list(test_examples)}}
        return experiment_data

    def get_data_for_experiment(self, domain: str, regime: Optional[str] = None) -> dict:
        """Load the data folds following the structure used in the experiments

        https://arxiv.org/pdf/2204.13021.pdf

        Args:
            domain: (str) 'banking', 'hotels', 'all', 'hotels-banking' or 'banking-hotels
            regime: (str) 'low', 'mid' or 'large' (or None for cross domain experiments)

        Returns:
            Dict with the folds ready for the experiment
        """
        if domain in [_HOTELS_BANKING, _BANKING_HOTELS]:
            source_domain, target_domain = domain.split("-")
            return self._get_cross_domain_data(source_domain, target_domain)

        assert regime in [_LOW, _MID, _LARGE], (
            "regime must be 'low', 'mid', 'large'")
        assert domain in [_BANKING, _HOTELS, _ALL], (
            "regime must be 'banking', 'hotels', 'all', 'hotels-banking' or 'banking-hotels'")
        
        if regime == _LOW:
            folds = range(20)
        else:
            folds = range(0, 20, 2)
        experiment_data = {}

        for fold_i in folds:
            if regime == _LOW:
                train_folds = [fold_i]
            elif regime == _MID:
                train_folds = [fold_i, fold_i + 1]
            else:
                train_folds = [j for j in range(20) if j not in [fold_i, fold_i+1]]

            test_folds = [j for j in range(20) if j not in train_folds]
            train_examples, test_examples = [], []

            for fold_j in train_folds:
                if domain in [_BANKING, _ALL]:
                    train_examples += [self._process_sample(sample) for sample in self._data[_BANKING][fold_j]]
                if domain in [_HOTELS, _ALL]:
                    train_examples += [self._process_sample(sample) for sample in self._data[_HOTELS][fold_j]]

            for fold_j in test_folds:
                if domain in [_BANKING, _ALL]:
                    test_examples += [self._process_sample(sample) for sample in self._data[_BANKING][fold_j]]
                if domain in [_HOTELS, _ALL]:
                    test_examples += [self._process_sample(sample) for sample in self._data[_HOTELS][fold_j]]

            fold_key = fold_i if regime == _LOW else fold_i / 2
            experiment_data[fold_key] = {"train": Dataset.from_list(train_examples), 
                                         "test": Dataset.from_list(test_examples)}

        return experiment_data
