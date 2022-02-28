import os

from dataloader.classifier import InputExample
from dataprocessor.data_processor import DataProcessor


class ClassifierProcessor(DataProcessor):

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            input_ids = line[0]
            word_lengths = line[1]
            word_lengths2 = line[2]
            word_lengths3 = line[3]
            label = line[4]
            examples.append(
                InputExample(guid=guid, input_ids=input_ids, word_lengths_1=word_lengths, word_lengths_2=word_lengths2,
                             word_lengths_3=word_lengths3, label=label))
        return examples



def get_processor(labels):
    ClassifierProcessor.get_labels = lambda _: labels
    return ClassifierProcessor
