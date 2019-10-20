import itertools
import json
import pickle
import logging
from io import TextIOWrapper
from typing import Dict
from ipdb import set_trace as bp

import torch
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (ArrayField, Field, LabelField, ListField,
                                  MetadataField, TextField)
from allennlp.data.instance import Instance
from overrides import overrides

from neural_persona.common.util import load_sparse, load_named_sparse

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("vampire_context_reader")
class VampireContextReader(DatasetReader):
    """ 
    Reads bag of word vectors from a sparse matrices representing training and validation data.

    Expects a sparse matrix of size N documents x vocab size, which can be created via 
    the scripts/preprocess_data.py file.

    The output of ``read`` is a list of ``Instances`` with the field:
        vec: ``ArrayField``

    Parameters
    ----------
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self, lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

    @overrides
    def _read(self, file_path):
        examples = np.load(file_path)
        for ix, example in enumerate(examples):
            instance = self.text_to_instance(example)
            # bp()
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(self, vec: np.array) -> Instance:  # type: ignore
        """
        Parameters
        ----------
        vec : ``np.array``, required.
            The text to classify

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields['tokens'] = ArrayField(vec)
        return Instance(fields)
