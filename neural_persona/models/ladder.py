import logging
import os
from functools import partial
from itertools import combinations
from operator import is_not
from typing import Any, Dict, List, Optional, Tuple, Union
from ipdb import set_trace as bp
import seaborn as sns
import numpy as np
from collections import Counter
import pandas
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides
from scipy import sparse
from tabulate import tabulate
from torch.nn.functional import log_softmax

from neural_persona.common.util import (compute_background_log_frequency, load_sparse,
                                        read_json)
from neural_persona.modules import VAE, LadderVAE
from neural_persona.modules.encoder import Encoder
from neural_persona.common.util import create_trainable_BatchNorm1d

logger = logging.getLogger(__name__)


@Model.register("ladder")
class Ladder(Model):
    """
    avitm is a pytorch reimplementation of [https://github.com/akashgit/autoencoding_vi_for_topic_models]

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
                 apply_batchnorm_on_recon: bool = False,
                 batchnorm_weight_learnable: bool = False,
                 batchnorm_bias_learnable: bool = True,
                 kl_weight_annealing: str = "constant",
                 linear_scaling: float = 1000.0,
                 sigmoid_weight_1: float = 0.25,
                 sigmoid_weight_2: float = 15,
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 use_doc_info: str = False,
                 use_background: str = False,
                 background_data_path: str = None,
                 update_background_freq: bool = False,
                 track_topics: bool = True,
                 track_persona: bool = True,
                 track_npmi: bool = True,
                 visual_persona: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.metrics = {'nkld': Average(), 'nll': Average(), 'perp': Average()}

        self.vocab = vocab
        self.vae = vae
        self.track_topics = track_topics
        self.track_npmi = track_npmi
        self.track_persona = track_persona
        self.visual_persona = visual_persona
        self.vocab_namespace = "ladder"
        self._update_background_freq = update_background_freq

        vocab_size = self.vocab.get_vocab_size(self.vocab_namespace)
        self._use_doc_info = use_doc_info
        # bp()
        if use_doc_info:
            self.interpolation = torch.nn.Parameter(torch.zeros(2, requires_grad=True))
        self._background_freq = self.initialize_bg_from_file(file_=background_data_path) if use_background else 0
        print(self._background_freq)
        # bp()
        self._ref_counts = reference_counts

        if reference_vocabulary is not None:
            # Compute data necessary to compute NPMI every epoch
            logger.info("Loading reference vocabulary.")
            self._ref_vocab = read_json(cached_path(reference_vocabulary))
            self._ref_vocab_index = dict(zip(self._ref_vocab, range(len(self._ref_vocab))))
            logger.info("Loading reference count matrix.")
            self._ref_count_mat = load_sparse(cached_path(self._ref_counts))
            logger.info("Computing word interaction matrix.")
            self._ref_doc_counts = (self._ref_count_mat > 0).astype(float)
            self._ref_interaction = self._ref_doc_counts.T.dot(self._ref_doc_counts)
            self._ref_doc_sum = np.array(self._ref_doc_counts.sum(0).tolist()[0])
            logger.info("Generating npmi matrices.")
            (self._npmi_numerator,
             self._npmi_denominator) = self.generate_npmi_vals(self._ref_interaction,
                                                               self._ref_doc_sum)
            self.n_docs = self._ref_count_mat.shape[0]

        self._bag_of_words_embedder = bow_embedder

        self._kl_weight_annealing = kl_weight_annealing

        self._linear_scaling = float(linear_scaling)
        self._sigmoid_weight_1 = float(sigmoid_weight_1)
        self._sigmoid_weight_2 = float(sigmoid_weight_2)
        if kl_weight_annealing == "linear":
            self._kld_weight = min(1.0, 1 / self._linear_scaling)
        elif kl_weight_annealing == "sigmoid":
            self._kld_weight = float(1 / (1 + np.exp(-self._sigmoid_weight_1 * (1 - self._sigmoid_weight_2))))
        elif kl_weight_annealing == "constant":
            self._kld_weight = 1.0
        else:
            raise ConfigurationError("anneal type {} not found".format(kl_weight_annealing))

        # setup batchnorm
        self._apply_batchnorm_on_recon = apply_batchnorm_on_recon
        if apply_batchnorm_on_recon:
            self.bow_bn = create_trainable_BatchNorm1d(vocab_size,
                                                       weight_learnable=batchnorm_weight_learnable,
                                                       bias_learnable=batchnorm_bias_learnable,
                                                       eps=0.001, momentum=0.001, affine=True)

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
                # print(self._kld_weight)
                self._kl_epoch_tracker = _epoch_num
                self._cur_epoch += 1
                if self._kl_weight_annealing == "linear":
                    self._kld_weight = min(1.0, self._cur_epoch / self._linear_scaling)
                elif self._kl_weight_annealing == "sigmoid":
                    self._kld_weight = float(
                        1 / (1 + np.exp(- self._sigmoid_weight_1 * (self._cur_epoch - self._sigmoid_weight_2))))
                elif self._kl_weight_annealing == "constant":
                    self._kld_weight = 1.0
                else:
                    raise ConfigurationError("anneal type {} not found".format(self._kl_weight_annealing))

    def compute_custom_metrics_once_per_epoch(self, epoch_num: Optional[List[int]]) -> None:
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
                beta = torch.softmax(self.vae.get_beta(level="p"), dim=1)
                topics = self.extract_topics(beta, k=k)
                topic_table = tabulate(topics, headers=["Topic #", "Words"])
                topic_dir = os.path.join(os.path.dirname(self.vocab.serialization_dir), "topics")
                if not os.path.exists(topic_dir):
                    os.mkdir(topic_dir)
                ser_dir = os.path.dirname(self.vocab.serialization_dir)
                topic_filepath = os.path.join(ser_dir, "topics", "topics_{}.txt".format(epoch_num[0]))
                words = list(itertools.chain(*[words for _, words in topics[1:]]))

                if self.visual_persona:
                    top_k = 100
                    width = top_k // 3
                    topic_filepath_png = os.path.join(ser_dir, "topics",
                                                      "topics_{}_top_{}.png".format(self._metric_epoch_tracker, top_k))
                    word2count = Counter(words)
                    top_k_idx2count = dict(sorted(word2count.items(), key=lambda x: x[1], reverse=True)[:top_k])
                    df = pandas.DataFrame.from_dict(top_k_idx2count, orient='index')
                    ax = df.plot(kind='bar')
                    ax.tick_params(axis="x", labelsize=6)
                    figure = ax.get_figure()
                    figure.set_figheight(6)
                    figure.set_figwidth(width)
                    figure.subplots_adjust(bottom=0.7)
                    # figure.set_fontsize(4)
                    figure.savefig(topic_filepath_png, dpi=300)
                    figure.clf()
                with open(topic_filepath, 'w+') as file_:
                    file_.write(topic_table)

            if self.track_npmi:
                if self._ref_vocab:
                    topics = self.extract_topics(self.vae.get_beta(level="p"))
                    self._cur_npmi = self.compute_npmi(topics[1:])
            self._metric_epoch_tracker = epoch_num[0]

    def extract_topics(self, weights: torch.Tensor, k: int = 20) -> List[Tuple[str, List[int]]]:
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
        interactions: ``np.ndarray``reference_vocabulary
            Interaction matrix of size reference vocab size x reference vocab size,
            where cell [i][j] indicates how many times word i and word j co-occur
            in the corpus.
        document_sums: ``np.ndarray``
            Matrix of size number of docs x reference vocab size, where
            cell [i][j] indicates how many times word i occur in documents
            in the corpus
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

        # For easy transfer to the GPU.
        self.device = self.vae.get_beta(level="p").device  # pylint: disable=W0201
        # self.device = self.vae.get_beta(level="t").device  # pylint: disable=W0201
        # bp()
        output_dict = {}

        if not self.training:
            self._kld_weight = 1.0  # pylint: disable=W0201
        else:
            self.update_kld_weight(epoch_num)
        # bp()
        # if you supply input as token IDs, embed them into bag-of-word-counts with a token embedder
        if isinstance(tokens, dict):
            embedded_tokens = self._bag_of_words_embedder(tokens['tokens']).to(device=self.device)
        else:
            embedded_tokens = tokens

        _, num_p, x_dim = embedded_tokens.shape
        if self._use_doc_info:
            # bp()
            embedded_doc_tokens, embedded_entity_tokens = embedded_tokens.split(x_dim // 2, dim=1)
            weights = torch.softmax(self.interpolation, dim=0)
            embedded_tokens = weights[0] * embedded_doc_tokens + weights[1] * embedded_entity_tokens
        else:
            # bp()
            assert x_dim == self.vocab.get_vocab_size(self.vocab_namespace) 
        # Encode the text into a shared representation for both the VAE

        # Perform variational inference.
        variational_output = self.vae(embedded_tokens)

        # Reconstructed bag-of-words from the VAE with background bias.
        reconstructed_bow = variational_output['reconstruction'] + self._background_freq

        # Apply batch_norm to the reconstructed bag of words.
        # Helps with word variety in topic space.

        reconstructed_bow = self.bow_bn(reconstructed_bow) if self._apply_batchnorm_on_recon else reconstructed_bow

        # Reconstruction log likelihood: log P(x | z) = log softmax(z beta + b)
        if self._use_doc_info:
            reconstruction_loss = self.bow_reconstruction_loss(reconstructed_bow, embedded_entity_tokens)
        else:
            reconstruction_loss = self.bow_reconstruction_loss(reconstructed_bow, embedded_tokens)

        # KL-divergence that is returned is the mean of the batch by default.
        negative_kl_divergence = variational_output['negative_kl_divergence']

        # Compute ELBo
        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss

        loss = -torch.mean(elbo)

        output_dict['loss'] = loss
        theta_t = variational_output['theta_t']
        theta_p = variational_output['theta_p']

        # Keep track of internal states for use downstream
        activations: List[Tuple[str, torch.FloatTensor]] = []

        activations.append(('theta_t', theta_t))
        activations.append(('theta_p', theta_p))

        output_dict['activations'] = activations

        # Update metrics
        nkld = -torch.mean(negative_kl_divergence)
        nll = -torch.mean(reconstruction_loss)
        if torch.isnan(nkld):
            bp()
        if torch.isnan(nll):
            bp()
        if torch.isnan(loss):
            bp()
        
        self.metrics['nkld'](nkld)
        self.metrics['nll'](nll)
        self.metrics['perp'](loss)

        # batch_num is tracked for kl weight annealing
        self.batch_num += 1

        self.compute_custom_metrics_once_per_epoch(epoch_num)

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
