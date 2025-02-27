import logging
import os
from functools import partial
from itertools import combinations
from operator import is_not
from typing import Dict, List, Optional, Tuple, Union
from ipdb import set_trace as bp
import numpy as np
import seaborn as sns
import torch
import itertools
from collections import Counter
import pandas

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average
from overrides import overrides
from scipy import sparse
from tabulate import tabulate
from torch.nn.functional import log_softmax

from neural_persona.common.util import (compute_background_log_frequency, load_sparse,
                                 read_json)
from neural_persona.modules import VAE

logger = logging.getLogger(__name__)

def print_param_for_check(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(f"name: {name}")
        print(f"param sum: {param.sum()}")
        print(f"param abs sum: {param.abs().sum()}")
        print(f"param abs max: {param.abs().max()}")
        if param.grad is not None:
            print(f"param grad sum: {param.grad.sum()}")
            print(f"param grad abs sum: {param.grad.abs().sum()}")
            print(f"param grad abs max: {param.grad.abs().max()}")
        print()
    print("-" * 80)


@Model.register("leo")
class Leo(Model):
    """
    VAMPIRE is a variational document model for pretraining under low
    resource environments.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    bow_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model
        into a bag-of-word-counts.
    vae : ``VAE``, required
        The variational autoencoder used to project the BoW into a latent space.
    kl_weight_annealing : ``string``, required
        Annealing weight on the KL divergence of ELBO.
        Choice between `sigmoid`, `linear` and `constant` annealing.
    linear_scaling: ``float``
        scaling applied ot KL weight annealing
    sigmoid_weight_1: ``float``
        first weight applied to sigmoid KL annealing
    sigmoid_weight_2: ``float``
        second weight applied to sigmoid KL annealing
    background_data_path: ``str``
        Path to a JSON file containing word frequencies accumulated over the training corpus.
    reference_counts: ``str``
        Path to reference counts for NPMI calculation
    reference_vocabulary: ``str``
        Path to reference vocabulary for NPMI calculation
    update_background_freq: ``bool``:
        Whether to allow the background frequency to be learnable.
    track_topics: ``bool``:
        Whether to periodically print the learned topics.
    track_npmi: ``bool``:
        Whether to track NPMI every epoch.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 bow_embedder: TokenEmbedder,
                 vae: VAE,
                 kl_weight_annealing: str = "constant",
                 linear_scaling: float = 1000.0,
                 sigmoid_weight_1: float = 0.25,
                 sigmoid_weight_2: float = 15,
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 background_data_path: str = None,
                 update_background_freq: bool = False,
                 track_topics: bool = True,
                 track_npmi: bool = True,
                 visual_topic: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.metrics = {'nkld': Average(), 'doc_nkld': Average(), 'entity_nkld': Average(), 'nll': Average()}

        self.vocab = vocab
        self.vae = vae
        self.track_topics = track_topics
        self.track_npmi = track_npmi
        self.visual_topic = visual_topic
        self.vocab_namespace = "entity_based"
        self._update_background_freq = update_background_freq
        # bp()
        self._background_freq = self.initialize_bg_from_file(file_=background_data_path)
        # bp()
        self._ref_counts = reference_counts
        self._npmi_updated = False

        if reference_vocabulary is not None:
            # Compute data necessary to compute NPMI every epoch
            logger.info("Loading reference vocabulary.")
            self._ref_vocab = read_json(cached_path(reference_vocabulary))
            self._ref_vocab_index = dict(zip(self._ref_vocab, range(len(self._ref_vocab))))
            logger.info("Loading reference count matrix.")
            self._ref_count_mat = load_sparse(cached_path(self._ref_counts))
            logger.info("Computing word interaction matrix.")
            self._ref_doc_counts = (self._ref_count_mat > 0).astype(float)
            self._ref_interaction = (self._ref_doc_counts).T.dot(self._ref_doc_counts)
            self._ref_doc_sum = np.array(self._ref_doc_counts.sum(0).tolist()[0])
            logger.info("Generating npmi matrices.")
            (self._npmi_numerator,
             self._npmi_denominator) = self.generate_npmi_vals(self._ref_interaction,
                                                               self._ref_doc_sum)
            self.n_docs = self._ref_count_mat.shape[0]

        vocab_size = self.vocab.get_vocab_size(self.vocab_namespace)
        self._bag_of_words_embedder = bow_embedder

        self._kl_weight_annealing = kl_weight_annealing

        self._linear_scaling = float(linear_scaling)
        self._sigmoid_weight_1 = float(sigmoid_weight_1)
        self._sigmoid_weight_2 = float(sigmoid_weight_2)
        if kl_weight_annealing == "linear":
            self._kld_weight = min(1, 1 / self._linear_scaling)
        elif kl_weight_annealing == "sigmoid":
            self._kld_weight = float(1/(1 + np.exp(-self._sigmoid_weight_1 * (1 - self._sigmoid_weight_2))))
        elif kl_weight_annealing == "constant":
            self._kld_weight = 1.0
        else:
            raise ConfigurationError("anneal type {} not found".format(kl_weight_annealing))

        # setup batchnorm
        self.doc_bow_bn = torch.nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.doc_bow_bn.weight.data.copy_(torch.ones(vocab_size, dtype=torch.float64))
        self.doc_bow_bn.weight.requires_grad = False

        # self.entity_bow_bn = torch.nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        # self.entity_bow_bn.weight.data.copy_(torch.ones(vocab_size, dtype=torch.float64))
        # self.entity_bow_bn.weight.requires_grad = False

        # Maintain these states for periodically printing topics and updating KLD
        self._metric_epoch_tracker = 0
        self._kl_epoch_tracker = 0
        self._cur_epoch = 0
        self._cur_npmi = 0.0
        self.batch_num = 0

        initializer(self)

    def initialize_bg_from_file(self, file_: Optional[str] = None) -> torch.Tensor:
        """
        Initialize the background frequency parameter from a file

        Parameters
        ----------
        ``file`` : str
            path to background frequency file
        """
        background_freq = compute_background_log_frequency(self.vocab, self.vocab_namespace, file_)
        return torch.nn.Parameter(background_freq, requires_grad=self._update_background_freq)

    @staticmethod
    def bow_reconstruction_loss(reconstructed_bow: torch.Tensor,
                                target_bow: torch.Tensor) -> torch.Tensor:
        """
        Initialize the background frequency parameter from a file

        Parameters
        ----------
        ``reconstructed_bow`` : torch.Tensor
            reconstructed bag of words from VAE
        ``target_bow`` : torch.Tensor
            target bag of words tensor

        Returns
        -------
        ``reconstruction_loss``
            Cross entropy loss between reconstruction and target
        """
        log_reconstructed_bow = log_softmax(reconstructed_bow + 1e-10, dim=-1)
        reconstruction_loss = torch.sum(target_bow * log_reconstructed_bow, dim=-1)
        return reconstruction_loss

    def update_kld_weight(self, epoch_num: Optional[List[int]]) -> None:
        """
        KL weight annealing scheduler

        Parameters
        ----------
        ``epoch_num`` : List[int]
            epoch tracker output (containing current epoch number)
        """
        if not epoch_num:
            self._kld_weight = 1.0
        else:
            _epoch_num = epoch_num[0]
            if _epoch_num != self._kl_epoch_tracker:
                print(self._kld_weight)
                self._kl_epoch_tracker = _epoch_num
                self._cur_epoch += 1
                if self._kl_weight_annealing == "linear":
                    self._kld_weight = min(1, self._cur_epoch / self._linear_scaling)
                elif self._kl_weight_annealing == "sigmoid":
                    self._kld_weight = float(1 / (1 + np.exp(- self._sigmoid_weight_1 * (self._cur_epoch - self._sigmoid_weight_2))))
                elif self._kl_weight_annealing == "constant":
                    self._kld_weight = 1.0
                else:
                    raise ConfigurationError("anneal type {} not found".format(self._kl_weight_annealing))

    def update_topics(self, epoch_num: Optional[List[int]]) -> None:
        """
        Update topics and NPMI once per epoch

        Parameters
        ----------
        ``epoch_num`` : List[int]
            epoch tracker output (containing current epoch number)
        """

        if epoch_num and epoch_num[0] != self._metric_epoch_tracker:

            # Logs the newest set of topics.
            if self.track_topics:
                k = 20
                # (K, vocabulary size)
                beta = torch.softmax(self.vae.get_beta(), dim=1)
                topics = self.extract_weights(beta, k=k)
                topic_table = tabulate(topics, headers=["Topic #", "Words"])
                topic_dir = os.path.join(os.path.dirname(self.vocab.serialization_dir), "topics")
                if not os.path.exists(topic_dir):
                    os.mkdir(topic_dir)
                ser_dir = os.path.dirname(self.vocab.serialization_dir)
                # bp()
                # Topics are saved for the previous epoch.
                topic_filepath = os.path.join(ser_dir, "topics", "topics_{}.txt".format(self._metric_epoch_tracker))
                with open(topic_filepath, 'w+') as file_:
                    file_.write(topic_table)

                W = torch.softmax(self.vae.get_W(), dim=1)
                personas = self.extract_weights(W, k=k)
                persona_table = tabulate(personas, headers=["Persona #", "Words"])
                persona_dir = os.path.join(os.path.dirname(self.vocab.serialization_dir), "personas")
                if not os.path.exists(persona_dir):
                    os.mkdir(persona_dir)
                ser_dir = os.path.dirname(self.vocab.serialization_dir)
                # bp()
                # Topics are saved for the previous epoch.
                persona_filepath = os.path.join(ser_dir, "personas",
                                                "personas_{}.txt".format(self._metric_epoch_tracker))
                with open(persona_filepath, 'w+') as file_:
                    file_.write(persona_table)

            self._metric_epoch_tracker = epoch_num[0]

    def update_npmi(self) -> None:
        """
        Update topics and NPMI at the beginning of validation.

        Parameters
        ----------
        ``epoch_num`` : List[int]
            epoch tracker output (containing current epoch number)
        """

        if self.track_npmi and self._ref_vocab and not self.training and not self._npmi_updated:
            topics = self.extract_weights(self.vae.get_beta())
            self._cur_npmi = self.compute_npmi(topics[1:])
            self._npmi_updated = True
        elif self.training:
            self._npmi_updated = False

    def extract_weights(self, weights: torch.Tensor, k: int = 20) -> List[Tuple[str, List[int]]]:
        """
        Given the learned (K, vocabulary size) weights, print the
        top k words from each row as a topic.

        Parameters
        ----------
        weights: ``torch.Tensor``
            The weight matrix whose second dimension equals the vocabulary size.
        k: ``int``
            The number of words per topic to display.

        Returns
        -------
        topics: ``List[Tuple[str, List[int]]]``
            collection of learned topics
        """

        words = list(range(weights.size(1)))
        words = [self.vocab.get_token_from_index(i, self.vocab_namespace) for i in words]

        topics = []

        word_strengths = list(zip(words, self._background_freq.tolist()))
        sorted_by_strength = sorted(word_strengths,
                                    key=lambda x: x[1],
                                    reverse=True)
        background = [x[0] for x in sorted_by_strength][:k]
        topics.append(('bg', background))

        for i, topic in enumerate(weights):
            word_strengths = list(zip(words, topic.tolist()))
            sorted_by_strength = sorted(word_strengths,
                                        key=lambda x: x[1],
                                        reverse=True)
            top_k = [x[0] for x in sorted_by_strength][:k]
            topics.append((str(i), top_k))

        return topics

    @staticmethod
    def generate_npmi_vals(interactions, document_sums):
        """
        Compute npmi values from interaction matrix and document sums

        Parameters
        ----------
        interactions: ``np.ndarray``
            Interaction matrix of size reference vocab size x reference vocab size,
            where cell [i][j] indicates how many times word i and word j co-occur
            in the corpus.
        document_sums: ``np.ndarray``
            Matrix of size number of docs x reference vocab size, where
            cell [i][j] indicates how many times word i occur in documents
            in the corpus
        TODO(suchin): update this documentation
        """
        interaction_rows, interaction_cols = interactions.nonzero()
        logger.info("generating doc sums...")
        doc_sums = sparse.csr_matrix((np.log10(document_sums[interaction_rows])
                                      + np.log10(document_sums[interaction_cols]),
                                      (interaction_rows, interaction_cols)),
                                     shape=interactions.shape)
        logger.info("generating numerator...")
        interactions.data = np.log10(interactions.data)
        numerator = interactions - doc_sums
        logger.info("generating denominator...")
        denominator = interactions
        return numerator, denominator

    def compute_npmi(self, topics, num_words=10):
        """
        Compute global NPMI across topics

        Parameters
        ----------
        topics: ``List[Tuple[str, List[int]]]``
            list of learned topics
        num_words: ``int``
            number of words to compute npmi over
        """
        topics_idx = [[self._ref_vocab_index.get(word)
                       for word in topic[1][:num_words]] for topic in topics]
        rows = []
        cols = []
        res_rows = []
        res_cols = []
        max_seq_len = max([len(topic) for topic in topics_idx])

        for index, topic in enumerate(topics_idx):
            topic = list(filter(partial(is_not, None), topic))
            if len(topic) > 1:
                _rows, _cols = zip(*combinations(topic, 2))
                res_rows.extend([index] * len(_rows))
                res_cols.extend(range(len(_rows)))
                rows.extend(_rows)
                cols.extend(_cols)

        npmi_data = ((np.log10(self.n_docs) + self._npmi_numerator[rows, cols])
                     / (np.log10(self.n_docs) - self._npmi_denominator[rows, cols]))
        npmi_data[npmi_data == 1.0] = 0.0
        npmi_shape = (len(topics), len(list(combinations(range(max_seq_len), 2))))
        npmi = sparse.csr_matrix((npmi_data.tolist()[0], (res_rows, res_cols)), shape=npmi_shape)
        return npmi.mean()

    def freeze_weights(self) -> None:
        """
        Freeze the weights of the VAE.
        """
        model_parameters = dict(self.vae.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Union[Dict[str, torch.IntTensor], torch.IntTensor],
                entities: Union[Dict[str, torch.IntTensor], torch.IntTensor],
                epoch_num: List[int] = None):
        """
        Parameters
        ----------
        tokens: ``Union[Dict[str, torch.IntTensor], torch.IntTensor]``
            A batch of tokens. We expect tokens to be represented in one of two ways:
                1. As token IDs. This representation will be used with downstream models, where bag-of-word count embedding
                must be done on the fly. If token IDs are provided, we use the bag-of-word-counts embedder to embed these
                tokens during training.
                2. As pre-computed bag of words vectors. This representation will be used during pretraining, where we can
                precompute bag-of-word counts and train much faster.
        epoch_num: ``List[int]``
            Output of epoch tracker
        """
        if self.batch_num in []:
            bp()
        # For easy transfer to the GPU.
        self.device = self.vae.get_beta().device  # pylint: disable=W0201

        output_dict = {}

        self.update_npmi()
        self.update_topics(epoch_num)

        if not self.training:
            self._kld_weight = 1.0  # pylint: disable=W0201
        else:
            self.update_kld_weight(epoch_num)
        # bp()
        # if you supply input as token IDs, embed them into bag-of-word-counts with a token embedder
        if isinstance(entities, dict):
            embedded_entities = (self._bag_of_words_embedder(entities['tokens'])
                                 .to(device=self.device))
        else:
            embedded_entities = entities

        if isinstance(tokens, dict):
            embedded_docs = (self._bag_of_words_embedder(tokens['tokens'])
                             .to(device=self.device))
        else:
            embedded_docs = tokens
        # Encode the text into a shared representation for both the VAE
        # and downstream classifiers to use.
        # bp()
        variational_output = self.vae(embedded_docs, embedded_entities)
        entities_mask = (embedded_entities.sum(-1) != 0).float()
        # bp()
        # Reconstructed bag-of-words from the VAE with background bias.
        doc_reconstructed_bow = variational_output['doc_reconstruction'] + self._background_freq
        entity_reconstructed_bow = variational_output['entity_reconstruction'] + self._background_freq

        # Apply batchnorm to the reconstructed bag of words.
        # Helps with word variety in topic space.
        doc_reconstructed_bow = self.doc_bow_bn(doc_reconstructed_bow)
        # entity_reconstructed_bow = self.entity_bow_bn(entity_reconstructed_bow) * entities_mask.unsqueeze(-1)

        # Reconstruction log likelihood: log P(x | z) = log softmax(z beta + b)
        reconstruction_loss = self.bow_reconstruction_loss(doc_reconstructed_bow, embedded_docs)
        # bp()
        reconstruction_loss += (self.bow_reconstruction_loss(entity_reconstructed_bow, embedded_entities) * entities_mask).sum(1)

        # KL-divergence that is returned is the mean of the batch by default.
        doc_negative_kl_divergence = variational_output['doc_negative_kl_divergence']
        # masked sum of entity KL-divergence since there are some paddings
        entity_negative_kl_divergence = torch.sum(variational_output["entity_negative_kl_divergence"] * entities_mask, dim=-1)
        # total KL-divergence is the sum of doc's KL and entities' KL
        negative_kl_divergence = doc_negative_kl_divergence  #  + entity_negative_kl_divergence
        # Compute ELBO
        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss

        loss = -torch.mean(elbo)
        open(f"{self.vae._get_name()}_loss.txt", "a+").write(f"{loss} \n")
        if torch.isnan(loss):
            bp()
        output_dict['loss'] = loss
        theta = variational_output['theta']

        # Keep track of internal states for use downstream
        activations: List[Tuple[str, torch.FloatTensor]] = []

        activations.append(('theta', theta))

        output_dict['activations'] = activations
        # bp()
        # Update metrics
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['doc_nkld'](-torch.mean(doc_negative_kl_divergence))
        self.metrics['entity_nkld'](-torch.mean(entity_negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))
        # bp()
        # batch_num is tracked for kl weight annealing
        self.batch_num += 1

        self.metrics['npmi'] = self._cur_npmi

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output
