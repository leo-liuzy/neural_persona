from typing import Dict
import warnings
import os
import torch
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from ipdb import set_trace as bp
from overrides import overrides

from neural_persona.common.util import normal_kl, create_trainable_BatchNorm1d, EPSILON
from allennlp.models import Model
from neural_persona.modules.vae.vae import VAE

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

@VAE.register("basic-l")
class BasicLVAE(VAE):
    """
    A Ladder Variational Autoencoder with 2 hidden layer and a Normal prior. This is a generalization of LogitNormal
    So far this class support:
        - normal prior with any simple mean and simple variance  --- "normal"
            (e.g. mu = [0, 0, ..., 0], var = [1, 1, ..., 1])
        - Laplacian Approximation of logit normal to Symmetric Dirichlet Distribution  --- "laplace-approx"
    """
    def __init__(self,
                 vocab,
                 encoder_topic: FeedForward,
                 mean_projection_topic: FeedForward,
                 log_variance_projection_topic: FeedForward,
                 encoder_entity: FeedForward,
                 mean_projection_entity: FeedForward,
                 log_variance_projection_entity: FeedForward,
                 encoder_entity_approx: FeedForward,
                 mean_projection_entity_approx: FeedForward,
                 log_variance_projection_entity_approx: FeedForward,
                 decoder_topic: FeedForward,  # decode topic input to persona hidden
                 decoder_mean_projection_topic: FeedForward,
                 decoder_log_variance_projection_topic: FeedForward,
                 decoder_persona: FeedForward,
                 prior: Dict = {"type": "normal", "mu": 0, "var": 1},
                 apply_batchnorm_on_normal: bool = False,
                 apply_batchnorm_on_decoder: bool = False,
                 batchnorm_weight_learnable: bool = False,
                 batchnorm_bias_learnable: bool = True,
                 stochastic_beta: bool = False,
                 z_dropout: float = 0.2) -> None:
        super(BasicLVAE, self).__init__(vocab)

        self.encoder_topic = encoder_topic
        self.mean_topic = mean_projection_topic
        self.log_var_topic = log_variance_projection_topic
        self.encoder_entity = encoder_entity
        self.mean_entity = mean_projection_entity
        self.log_var_entity = log_variance_projection_entity
        self.encoder_entity_approx = encoder_entity_approx
        self.mean_entity_approx = mean_projection_entity_approx
        self.log_var_entity_approx = log_variance_projection_entity_approx
        self.decoder_mean_topic = decoder_mean_projection_topic
        self.decoder_log_var_topic = decoder_log_variance_projection_topic
        self._decoder_topic = torch.nn.Linear(decoder_topic.get_input_dim(), decoder_topic.get_output_dim(),
                                              bias=False)
        self._decoder_persona = torch.nn.Linear(decoder_persona.get_input_dim(), decoder_persona.get_output_dim(),
                                                bias=False)
        self._z_dropout = torch.nn.Dropout(z_dropout)

        self.num_topic = encoder_topic.get_output_dim()
        self.num_persona = encoder_entity.get_output_dim()

        self.prior = prior
        self.p_params = None
        # self.p_mu, self.p_sigma, self.p_log_var = None, None, None
        self.initialize_prior(prior)

        # If specified, established batchnorm for both mean and log variance.
        self._apply_batchnorm_on_normal = apply_batchnorm_on_normal
        self.mean_bn_entity, self.log_var_bn_entity = None, None
        self.mean_bn_topic, self.log_var_bn_topic = None, None
        self.decoder_mean_bn_topic, self.decoder_log_var_bn_topic = None, None
        self.mean_bn_entity_approx, self.log_var_bn_entity_approx = None, None
        if apply_batchnorm_on_normal:
            self.mean_bn_topic = create_trainable_BatchNorm1d(self.num_topic,
                                                              weight_learnable=batchnorm_weight_learnable,
                                                              bias_learnable=batchnorm_bias_learnable,
                                                              eps=0.001, momentum=0.001, affine=True)
            self.log_var_bn_topic = create_trainable_BatchNorm1d(self.num_topic,
                                                                 weight_learnable=batchnorm_weight_learnable,
                                                                 bias_learnable=batchnorm_bias_learnable,
                                                                 eps=0.001, momentum=0.001, affine=True)
            self.decoder_mean_bn_topic = create_trainable_BatchNorm1d(self.num_persona,
                                                                      weight_learnable=batchnorm_weight_learnable,
                                                                      bias_learnable=batchnorm_bias_learnable,
                                                                      eps=0.001, momentum=0.001, affine=True)
            self.decoder_log_var_bn_topic = create_trainable_BatchNorm1d(self.num_persona,
                                                                         weight_learnable=batchnorm_weight_learnable,
                                                                         bias_learnable=batchnorm_bias_learnable,
                                                                         eps=0.001, momentum=0.001, affine=True)

        # If specified, established batchnorm for reconstruction matrix, applying batch norm across vocabulary
        self._apply_batchnorm_on_decoder = apply_batchnorm_on_decoder
        if apply_batchnorm_on_decoder:
            self.decoder_bn_topic = create_trainable_BatchNorm1d(decoder_topic.get_output_dim(),
                                                                 weight_learnable=batchnorm_weight_learnable,
                                                                 bias_learnable=batchnorm_bias_learnable,
                                                                 eps=0.001, momentum=0.001, affine=True)
            self.decoder_bn_persona = create_trainable_BatchNorm1d(decoder_persona.get_output_dim(),
                                                                   weight_learnable=batchnorm_weight_learnable,
                                                                   bias_learnable=batchnorm_bias_learnable,
                                                                   eps=0.001, momentum=0.001, affine=True)
        # If specified, constrain each topic to be a distribution over vocabulary
        self._stochastic_beta = stochastic_beta

    def initialize_prior(self, prior: Dict):
        if prior['type'] == "normal":
            if 'mu' not in prior or 'var' not in prior:
                raise Exception("MU, VAR undefined for normal")
            mu = torch.zeros(1, self.num_topic).fill_(prior['mu'])
            var = torch.zeros(1, self.num_topic).fill_(prior['var'])
            sigma = torch.sqrt(var)
            log_var = var.log()

        elif prior['type'] == "laplace-approx":
            a = torch.zeros(1, self.num_topic).fill_(prior['alpha'])
            mu = a.log() - torch.mean(a.log(), 1)
            var = 1.0 / a * (1 - 2.0 / self.num_topic) + 1.0 / self.num_topic * torch.mean(1 / a)
            sigma = torch.sqrt(var)
            log_var = var.log()
        else:
            raise Exception("Invalid/Undefined prior!")

        # parameters of prior distribution are not trainable
        self.register_buffer("p_mu", mu)
        self.register_buffer("p_sigma", sigma)
        self.register_buffer("p_log_var", log_var)
        # self.p_params = {
        #     "mean": self.p_mu,
        #     "sigma": self.p_sigma,
        #     "log_variance": self.p_log_var
        # }

    @overrides
    def forward(self, entity_vector: torch.FloatTensor):  # pylint: disable = W0221
        """
        Given the input representation, produces the reconstruction from theta
        as well as the negative KL-divergence, theta itself, and the parameters
        of the distribution.
        """
        output = {}
        batch_size, max_num_entity, _ = entity_vector.shape

        # prior -- N(0, 1)
        p_params = {"mean": self.p_mu,
                    "sigma": self.p_sigma,
                    "log_variance": self.p_log_var}

        # bp()
        # estimate persona in bottom-up direction
        hidden_s = self.encoder_entity(entity_vector)
        # TODO: bn on entity are not used. wonder: should we run a batch on all global entity representation
        tilde_s_params = self.estimate_params(hidden_s, self.mean_entity, self.log_var_entity,
                                              self.mean_bn_entity, self.log_var_bn_entity)
        tilde_s = self.reparameterize(tilde_s_params)

        # estimate topic in bottom-up direction
        # TODO: free for other function choices e.g. avg(.)
        global_s, _ = tilde_s.max(1)
        hidden_d = self.encoder_topic(global_s)
        d_params = self.estimate_params(hidden_d, self.mean_topic, self.log_var_topic,
                                        self.mean_bn_topic, self.log_var_bn_topic)
        d = self.reparameterize(d_params)
        theta = torch.softmax(d, dim=-1)
        output.update({"d": d,
                       "theta": theta,
                       "d_params": d_params,
                       "d_negative_kl_divergence": self.compute_negative_kld(q_params=d_params,
                                                                             p_params=p_params)
                       })

        # estimate persona in top-down direction
        theta = theta.unsqueeze(1)  # .repeat(1, max_num_entity, 1)
        t = self.encoder_entity_approx(theta)
        t_params = self.estimate_params(t, self.mean_entity_approx, self.log_var_entity_approx,
                                        self.mean_bn_entity_approx, self.log_var_bn_entity_approx)

        # merge top-down and bottom-up inference of persona
        persona_params = self.merge_normal(tilde_s_params, t_params)

        # generative model's persona params
        W = self._decoder_topic.weight.t()
        if self._apply_batchnorm_on_decoder:
            W = self.decoder_bn_persona(W)
        if self._stochastic_beta:
            W = torch.nn.functional.softmax(W, dim=1)
        # TODO: use d or theta
        p_persona_hidden = d @ W
        # bp()
        decoder_persona_params = self.estimate_params(p_persona_hidden, self.decoder_mean_topic,
                                                      self.decoder_log_var_topic,
                                                      self.decoder_mean_bn_topic, self.decoder_log_var_bn_topic)
        decoder_persona_params = {k: v.unsqueeze(1) for k, v in decoder_persona_params.items()}
        inferred_persona = self.reparameterize(persona_params)
        e = torch.softmax(inferred_persona, dim=-1)
        output.update({"s": inferred_persona,
                       "e": e,
                       "s_params": persona_params,
                       "s_negative_kl_divergence": self.compute_negative_kld(q_params=persona_params,
                                                                             p_params=decoder_persona_params)})

        beta = self._decoder_persona.weight.t()
        if self._apply_batchnorm_on_decoder:
            beta = self.decoder_bn_persona(beta)
        if self._stochastic_beta:
            beta = torch.nn.functional.softmax(beta, dim=1)
        e_reconstruction = e @ beta
        output["e_reconstruction"] = e_reconstruction
        # bp()
        return output

    @overrides
    def estimate_params(self,
                        input_repr: torch.FloatTensor,
                        mean_projection: FeedForward,
                        log_variance_projection: FeedForward,
                        mean_bn: torch.nn.BatchNorm1d = None,
                        log_var_bn: torch.nn.BatchNorm1d = None):
        """
        Estimate the parameters for the logistic normal.
        """
        mean = mean_projection(input_repr)  # pylint: disable=C0103
        log_var = log_variance_projection(input_repr)

        if mean_bn is not None and log_var_bn is not None:
            if len(input_repr.shape) == 2:
                mean = mean_bn(mean)  # pylint: disable=C0103
                log_var = log_var_bn(log_var)  # pylint: disable=C0103
            else:
                mean = mean_bn(mean.transpose(1, 2)).transpose(1, 2)  # pylint: disable=C0103
                log_var = log_var_bn(log_var.transpose(1, 2)).transpose(1, 2)  # pylint: disable=C0103

        sigma = torch.sqrt(torch.exp(log_var))  # log_var is actually log (variance^2).

        return {
                "mean": mean,
                "sigma": sigma,
                "log_variance": log_var
                }

    @overrides
    def compute_negative_kld(self, q_params: Dict, p_params: Dict):
        """
        Compute the closed-form solution for negative KL-divergence for Gaussians.
        """
        mu, log_var = q_params["mean"], q_params["log_variance"]  # pylint: disable=C0103
        # bp()
        negative_kl_divergence = -normal_kl((mu, log_var), (p_params["mean"], p_params["log_variance"]))
        # bp()
        return negative_kl_divergence

    @staticmethod
    def merge_normal(param_d, param_t):
        """
        Merge the parameter of bottom-up(d) normal distribution and top-down(t) normal distribution
        (using LVAE notation)
        :param param_d: bottom-up normal distribution parameter
        :param param_t: top-down normal distribution parameter
        :return: approximate posterior distribution q(z_i | z_{i+1}, x)
        """

        mu_d, log_var_d = param_d["mean"], param_d["log_variance"]
        mu_t, log_var_t = param_t["mean"], param_t["log_variance"]  # use the bottom-up pass notation
        precision_d, precision_t = 1 / torch.exp(log_var_d), 1 / torch.exp(log_var_t)

        # Merge distributions into a single new distribution
        mu = ((mu_d * precision_d) + (mu_t * precision_t)) / (precision_d + precision_t)

        var = 1 / (precision_d + precision_t)
        sigma = torch.sqrt(var)
        log_var = torch.log(var + EPSILON)

        return {
            "mean": mu,
            "sigma": sigma,
            "log_variance": log_var
        }

    def reparameterize(self, params):
        """z is the result of the reparameterization trick."""
        mu, sigma = params["mean"], params["sigma"]  # pylint: disable=C0103

        # Generate random noise and sample theta.
        # Shape: (batch, latent_dim)
        batch_size = params["mean"].size(0)
        latent_dim = params["mean"].size(-1)
        if len(params["mean"].shape) == 3:
            persona_size = params["mean"].size(1)
        # Enable reparameterization for training only.
        if self.training:
            seed = os.environ['SEED']
            torch.manual_seed(seed)
            # Seed all GPUs with the same seed if available.
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            epsilon = torch.randn(batch_size, latent_dim).to(device=mu.device)
            if len(params["mean"].shape) == 3:
                epsilon = torch.randn(batch_size, persona_size, latent_dim).to(device=mu.device)
            z = mu + sigma * epsilon  # pylint: disable=C0103
        else:
            z = mu  # pylint: disable=C0103
        return z

    @overrides
    def get_beta(self):
        return self._decoder_persona._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212

    def get_W(self):
        return self._decoder_topic._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212
