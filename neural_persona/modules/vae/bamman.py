from typing import Dict
import warnings
import os
import torch
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from ipdb import set_trace as bp
from overrides import overrides

from neural_persona.common.util import normal_kl, create_trainable_BatchNorm1d, EPSILON, multinomial_kl
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


@VAE.register("bamman")
class Bamman(VAE):
    """
    A Variational Autoencoder with 1 hidden layer and a Normal prior. This is a generalization of LogitNormal
    So far this class support:
        - normal prior with any simple mean and simple variance  --- "normal"
            (e.g. mu = [0, 0, ..., 0], var = [1, 1, ..., 1])
        - Laplacian Approximation of logit normal to Symmetric Dirichlet Distribution  --- "laplace-approx"
    """
    def __init__(self,
                 vocab,
                 encoder_entity: FeedForward,
                 encoder_entity_global: FeedForward,
                 decoder_type: FeedForward,  # (d_dim -> P)
                 mean_projection_type: FeedForward,
                 log_var_projection_type: FeedForward,
                 decoder_topic: FeedForward,  # (K -> V)
                 decoder_persona: FeedForward,  # (P -> K)
                 prior: Dict = {"type": "normal", "mu": 0, "var": 1},
                 pooling_layer: str = "max",
                 apply_batchnorm_on_normal: bool = False,
                 apply_batchnorm_on_decoder: bool = False,
                 batchnorm_weight_learnable: bool = False,
                 batchnorm_bias_learnable: bool = True,
                 stochastic_weight: bool = False,
                 z_dropout: float = 0.2) -> None:
        super(Bamman, self).__init__(vocab)

        self.encoder_entity = encoder_entity
        self.encoder_entity_global = encoder_entity_global
        self.mean_projection_type = mean_projection_type
        self.log_var_projection_type = log_var_projection_type

        self._decoder_type = torch.nn.Linear(decoder_type.get_input_dim(),
                                             decoder_type.get_output_dim(), bias=False)
        self._decoder_topic = torch.nn.Linear(decoder_topic.get_input_dim(), decoder_topic.get_output_dim(),
                                              bias=False)
        self._decoder_persona = torch.nn.Linear(decoder_persona.get_input_dim(), decoder_persona.get_output_dim(),
                                                bias=False)
        self._z_dropout = torch.nn.Dropout(z_dropout)

        self.num_type = decoder_type.get_input_dim()
        self.num_topic = decoder_persona.get_output_dim()
        self.num_persona = decoder_persona.get_input_dim()

        self.prior = prior
        if pooling_layer not in ["max", "sum", "mean"]:
            raise Exception("Undefined pooling function")
        self.pooling_func = pooling_layer
        self.pooling_layer = getattr(torch, pooling_layer)
        self.p_params = None
        # self.p_mu, self.p_sigma, self.p_log_var = None, None, None
        self.initialize_prior(prior)

        # If specified, established batchnorm for both mean and log variance.
        self._apply_batchnorm_on_normal = apply_batchnorm_on_normal
        self.mean_bn_type, self.log_var_bn_type = None, None
        if apply_batchnorm_on_normal:
            self.mean_bn_type = create_trainable_BatchNorm1d(self.num_type,
                                                             weight_learnable=batchnorm_weight_learnable,
                                                             bias_learnable=batchnorm_bias_learnable,
                                                             eps=0.001, momentum=0.001, affine=True)
            self.log_var_bn_type = create_trainable_BatchNorm1d(self.num_type,
                                                                weight_learnable=batchnorm_weight_learnable,
                                                                bias_learnable=batchnorm_bias_learnable,
                                                                eps=0.001, momentum=0.001, affine=True)

        # If specified, established batchnorm for reconstruction matrix, applying batch norm across vocabulary
        self._apply_batchnorm_on_decoder = apply_batchnorm_on_decoder
        if apply_batchnorm_on_decoder:
            self.decoder_bn_type = create_trainable_BatchNorm1d(decoder_type.get_output_dim(),
                                                                weight_learnable=batchnorm_weight_learnable,
                                                                bias_learnable=batchnorm_bias_learnable,
                                                                eps=0.001, momentum=0.001, affine=True)
            self.decoder_bn_topic = create_trainable_BatchNorm1d(decoder_topic.get_output_dim(),
                                                                 weight_learnable=batchnorm_weight_learnable,
                                                                 bias_learnable=batchnorm_bias_learnable,
                                                                 eps=0.001, momentum=0.001, affine=True)

            self.decoder_bn_persona = create_trainable_BatchNorm1d(decoder_persona.get_output_dim(),
                                                                   weight_learnable=batchnorm_weight_learnable,
                                                                   bias_learnable=batchnorm_bias_learnable,
                                                                   eps=0.001, momentum=0.001, affine=True)
        # If specified, constrain each topic to be a distribution over vocabulary
        self._stochastic_weight = stochastic_weight
        ##  bp()

    def initialize_prior(self, prior: Dict):
        if prior['type'] == "normal":
            if 'mu' not in prior or 'var' not in prior:
                raise Exception("MU, VAR undefined for normal")
            mu = torch.zeros(1, self.num_type).fill_(prior['mu'])
            var = torch.zeros(1, self.num_type).fill_(prior['var'])
            sigma = torch.sqrt(var)
            log_var = var.log()

        elif prior['type'] == "laplace-approx":
            a = torch.zeros(1, self.num_type).fill_(prior['alpha'])
            mu = a.log() - torch.mean(a.log(), 1)
            var = 1.0 / a * (1 - 2.0 / self.num_type) + 1.0 / self.num_type * torch.mean(1 / a)
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
        # bp()
        # get shape dim for later use
        batch_size, max_num_entity, _ = entity_vector.shape
        # prior -- N(0, 1)
        p_params = {"mean": self.p_mu.repeat(batch_size, 1),
                    "sigma": self.p_sigma.repeat(batch_size, 1),
                    "log_variance": self.p_log_var.repeat(batch_size, 1)
                    }
        
        # estimate persona in bottom-up direction
        s_tilde = self.encoder_entity(entity_vector)
        e_tilde = gumbel_softmax(s_tilde)
        g_tilde = self.pooling_layer(e_tilde, dim=1)
        # g_tilde = (batch_size, P)
        if self.pooling_func == "max":
            g_tilde = g_tilde[0]
        if g_tilde.shape[1] != self.encoder_entity_global.get_input_dim():
            bp()
        g_tilde_hidden = self.encoder_entity_global(g_tilde)
        type_params = self.estimate_params(g_tilde_hidden, self.mean_projection_type,
                                           self.log_var_projection_type, self.mean_bn_type,
                                           self.log_var_bn_type)
        # calculate for the distribution for document representation
        # estimate the intermediate document representation
        d = self.reparameterize(type_params)
        theta = self._z_dropout(d)
        theta = torch.softmax(theta, dim=-1)
        output.update({"theta": theta,
                       "type_params": type_params,
                       "type_negative_kl_divergence": self.compute_negative_kld(q_params=type_params,
                                                                                p_params=p_params)
                       })

        f = self._decoder_type.weight.t()
        if self._stochastic_weight:
            f = torch.nn.functional.softmax(f, dim=1)
        if self._apply_batchnorm_on_decoder:
            f = self.decoder_bn_topic(f)

        # (batch_size, num_type) -> (batch_size, P) = global persona representation
        g = theta @ f
        output["global_persona"] = g
        # decode type representation to persona representation
        # (batch_size, max_num_entity, P) -- equivalent to sampling from multinomial(n=1, p_1, ... p_P)
        persona_proportion = gumbel_softmax(g.unsqueeze(1).repeat(1, max_num_entity, 1))

        q_persona_params = {"logit": g_tilde}
        p_persona_params = {"logit": g}
        
        persona_proportion = self._z_dropout(persona_proportion)
        # bp()
        output.update({"persona": persona_proportion,
                       "persona_params": q_persona_params,
                       "persona_negative_kl_divergence": self.compute_negative_kld(q_params=q_persona_params,
                                                                                  p_params=p_persona_params,
                                                                                  type="multinomial")})
        # bp()
        # decode persona representation to topic representation
        W = self._decoder_persona.weight.t()
        if self._apply_batchnorm_on_decoder:
            W = self.decoder_bn_persona(W)
        if self._stochastic_weight:
            W = torch.nn.functional.softmax(W, dim=1)
        # bp()
        # persona_reconstruction = topic_proportion calculated from persona proportion
        persona_reconstruction = torch.softmax(persona_proportion @ W, dim=-1)
        output["persona_reconstruction"] = persona_reconstruction

        # decode topic representation(proportion) to distribution over word(unnormalized)
        beta = self._decoder_topic.weight.t()
        if self._apply_batchnorm_on_decoder:
            beta = self.decoder_bn_topic(beta)
        if self._stochastic_weight:
            beta = torch.nn.functional.softmax(beta, dim=1)
        # bp()
        bow_reconstruction = persona_reconstruction @ beta
        output["bow_reconstruction"] = bow_reconstruction

        return output

    @overrides
    def estimate_params(self,
                        input_repr: torch.FloatTensor,
                        mean_projection: FeedForward,
                        log_variance_projection: FeedForward,
                        mean_bn: torch.nn.BatchNorm1d = None,
                        log_var_bn: torch.nn.BatchNorm1d = None):
        """
        Estimate the parameters for the normal.
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
    def compute_negative_kld(self, q_params: Dict, p_params: Dict, type: str = "normal"):
        """
        Compute the closed-form solution for negative KL-divergence for Gaussians.
        Not averaged across batch
        """
        if type == "normal":
            mu, log_var = q_params["mean"], q_params["log_variance"]  # pylint: disable=C0103
            # bp()
            negative_kl_divergence = -normal_kl((mu, log_var), (p_params["mean"], p_params["log_variance"]))
            # bp()
        elif type == "multinomial":
            q_logit = q_params["logit"]
            p_logit = p_params["logit"]
            # bp()
            negative_kl_divergence = -multinomial_kl(q_logit, p_logit)
        else:
            raise Exception("Undefined distribution")
        return negative_kl_divergence

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
        return self._decoder_topic._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212

    def get_W(self):
        persona = self._decoder_persona._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212
        # topic = self._decoder_topic._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212
        return persona  # @ topic
