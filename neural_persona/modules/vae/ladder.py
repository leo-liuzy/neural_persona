from typing import Dict
import os
import torch
from allennlp.modules import FeedForward
from overrides import overrides

from neural_persona.common.util import normal_kl, create_trainable_BatchNorm1d, EPSILON
from neural_persona.modules.vae.vae import VAE


@VAE.register("ladder")
class LadderVAE(VAE):
    """
    A Variational Autoencoder with 1 hidden layer and a Normal prior. This is a generalization of LogitNormal
    So far this class support:
        - normal prior with any simple mean and simple variance  --- "normal"
            (e.g. mu = [0, 0, ..., 0], var = [1, 1, ..., 1])
        - Laplacian Approximation of logit normal to Symmetric Dirichlet Distribution  --- "laplace-approx"
    """
    def __init__(self,
                 vocab,
                 encoder_d1: FeedForward,
                 mean_projection_d1: FeedForward,
                 log_variance_projection_d1: FeedForward,
                 encoder_d2: FeedForward,
                 mean_projection_d2: FeedForward,
                 log_variance_projection_d2: FeedForward,
                 encoder_t1: FeedForward,
                 mean_projection_t1: FeedForward,
                 log_variance_projection_t1: FeedForward,
                 decoder1: FeedForward,
                 decoder2: FeedForward,
                 prior: Dict = {"type": "normal", "mu": 0, "var": 1},
                 apply_batchnorm_on_normal: bool = False,
                 apply_batchnorm_on_decoder: bool = False,
                 batchnorm_weight_learnable: bool = False,
                 batchnorm_bias_learnable: bool = True,
                 stochastic_beta: bool = False,
                 z_dropout: float = 0.2) -> None:
        super(LadderVAE, self).__init__(vocab)
        self.encoder_d1 = encoder_d1
        self.mean_projection_d1 = mean_projection_d1
        self.log_variance_projection_d1 = log_variance_projection_d1
        self.encoder_d2 = encoder_d2
        self.mean_projection_d2 = mean_projection_d2
        self.log_variance_projection_d2 = log_variance_projection_d2
        self.encoder_t1 = encoder_t1
        self.mean_projection_t1 = mean_projection_t1
        self.log_variance_projection_t1 = log_variance_projection_t1

        self._decoder1 = torch.nn.Linear(decoder1.get_input_dim(), decoder1.get_output_dim(),
                                         bias=False)
        self._decoder2 = torch.nn.Linear(decoder2.get_input_dim(), decoder2.get_output_dim(),
                                         bias=False)

        self._z_dropout = torch.nn.Dropout(z_dropout)

        self.num_persona = mean_projection_d1.get_output_dim()
        self.num_topic = mean_projection_d2.get_output_dim()

        self.prior = prior
        if prior['type'] == "normal":
            if 'mu' not in prior or 'var' not in prior:
                raise Exception("MU, VAR undefined for normal")
            p_mu = torch.zeros(1, self.num_topic).fill_(prior['mu'])
            p_var = torch.zeros(1, self.num_topic).fill_(prior['var'])
            p_log_var = p_var.log()

        elif prior['type'] == "laplace-approx":
            a = torch.zeros(1, self.num_topic).fill_(prior['alpha'])
            p_mu = a.log() - torch.mean(a.log(), 1)
            p_var = 1.0 / a * (1 - 2.0 / self.num_topic) + 1.0 / self.num_topic * torch.mean(1 / a)
            p_log_var = p_var.log()
        else:
            raise Exception("Invalid/Undefined prior!")

        # parameters of prior distribution are not trainable
        self.register_buffer("p_mu", p_mu)
        self.register_buffer("p_log_var", p_log_var)

        # If specified, established batchnorm for both mean and log variance.
        self._apply_batchnorm_on_normal = apply_batchnorm_on_normal
        self.mean_bn_d1, self.log_var_bn_d1 = None, None
        self.mean_bn_d2, self.log_var_bn_d2 = None, None
        self.mean_bn_t1, self.log_var_bn_t1 = None, None
        if apply_batchnorm_on_normal:
            self.mean_bn_d1 = create_trainable_BatchNorm1d(self.num_persona,
                                                           weight_learnable=batchnorm_weight_learnable,
                                                           bias_learnable=batchnorm_bias_learnable,
                                                           eps=0.001, momentum=0.001, affine=True)
            self.log_var_bn_d1 = create_trainable_BatchNorm1d(self.num_persona,
                                                              weight_learnable=batchnorm_weight_learnable,
                                                              bias_learnable=batchnorm_bias_learnable,
                                                              eps=0.001, momentum=0.001, affine=True)

            self.mean_bn_d2 = create_trainable_BatchNorm1d(self.num_topic,
                                                           weight_learnable=batchnorm_weight_learnable,
                                                           bias_learnable=batchnorm_bias_learnable,
                                                           eps=0.001, momentum=0.001, affine=True)
            self.log_var_bn_d2 = create_trainable_BatchNorm1d(self.num_topic,
                                                              weight_learnable=batchnorm_weight_learnable,
                                                              bias_learnable=batchnorm_bias_learnable,
                                                              eps=0.001, momentum=0.001, affine=True)
            self.mean_bn_t1 = create_trainable_BatchNorm1d(self.num_topic,
                                                           weight_learnable=batchnorm_weight_learnable,
                                                           bias_learnable=batchnorm_bias_learnable,
                                                           eps=0.001, momentum=0.001, affine=True)
            self.log_var_bn_t1 = create_trainable_BatchNorm1d(self.num_topic,
                                                              weight_learnable=batchnorm_weight_learnable,
                                                              bias_learnable=batchnorm_bias_learnable,
                                                              eps=0.001, momentum=0.001, affine=True)

        # If specified, established batchnorm for reconstruction matrix, applying batch norm across vocabulary
        self._apply_batchnorm_on_decoder = apply_batchnorm_on_decoder
        if apply_batchnorm_on_decoder:
            self.decoder_bn1 = create_trainable_BatchNorm1d(decoder1.get_output_dim(),
                                                            weight_learnable=batchnorm_weight_learnable,
                                                            bias_learnable=batchnorm_bias_learnable,
                                                            eps=0.001, momentum=0.001, affine=True)

            self.decoder_bn2 = create_trainable_BatchNorm1d(decoder2.get_output_dim(),
                                                            weight_learnable=batchnorm_weight_learnable,
                                                            bias_learnable=batchnorm_bias_learnable,
                                                            eps=0.001, momentum=0.001, affine=True)

        # If specified, constrain each topic to be a distribution over vocabulary
        self._stochastic_beta = stochastic_beta


    @overrides
    def forward(self, input_repr: torch.FloatTensor):  # pylint: disable = W0221
        """
        Given the input representation, produces the reconstruction from theta
        as well as the negative KL-divergence, theta itself, and the parameters
        of the distribution.
        """
        # bottom-up inference -- q(z_i | z_{i-1})
        d_1 = self.estimate_params(input_repr, self.mean_projection_d1, self.log_variance_projection_d1,
                                   self.mean_bn_d1, self.log_var_bn_d1)
        d_2 = self.estimate_params(d_1['mean'], self.mean_projection_d2, self.log_variance_projection_d2,
                                   self.mean_bn_d2, self.log_var_bn_d2)
        z_2 = self.reparameterize(params=d_2)
        t_1 = self.estimate_params(z_2, self.mean_projection_t1, self.log_variance_projection_t1,
                                   self.mean_bn_t1, self.log_var_bn_t1)
        z_1_params = self.merge_normal(param_d=d_1, param_t=t_1)


        theta = output["theta"]
        # self._decoder.weight (output_dim x input_dim)
        beta = self._decoder.weight.t()
        if self._apply_batchnorm_on_decoder:
            beta = self.decoder_bn(beta)
        if self._stochastic_beta:
            beta = torch.nn.functional.softmax(beta, dim=1)
        reconstruction = theta @ beta
        output["reconstruction"] = reconstruction

        return output

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
        log_var = torch.log(var + 1e-12)

        return {
                "mean": mu,
                "sigma": sigma,
                "log_variance": log_var
                }

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

        if self._apply_batchnorm_on_normal:
            mean = mean_bn(mean)  # pylint: disable=C0103
            log_var = log_var_bn(log_var)  # pylint: disable=C0103

        sigma = torch.sqrt(torch.exp(log_var))  # log_var is actually log (variance^2).

        return {
                "mean": mean,
                "sigma": sigma,
                "log_variance": log_var
                }

    @overrides
    def compute_negative_kld(self, params: Dict):
        """
        Compute the closed-form solution for negative KL-divergence for Gaussians.
        """
        mu, log_var = params["mean"], params["log_variance"]  # pylint: disable=C0103
        negative_kl_divergence = normal_kl((mu, log_var), (self.p_mu, self.p_log_var))
        return negative_kl_divergence

    def reparameterize(self, params):
        """z is the result of the reparameterization trick."""
        mu, sigma = params["mean"], params["sigma"]  # pylint: disable=C0103

        # Generate random noise and sample theta.
        # Shape: (batch, latent_dim)
        batch_size = params["mean"].size(0)

        # Enable reparameterization for training only.
        if self.training:
            seed = os.environ['SEED']
            torch.manual_seed(seed)
            # Seed all GPUs with the same seed if available.
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            epsilon = torch.randn(batch_size, self.latent_dim).to(device=mu.device)
            z = mu + sigma * epsilon  # pylint: disable=C0103
        else:
            z = mu  # pylint: disable=C0103
        return z

    @overrides
    def generate_latent_code(self, input_repr: torch.Tensor):
        """
        Given an input vector, produces the latent encoding z, followed by the
        mean and log variance of the variational distribution produced.

        z is the result of the reparameterization trick.
        (https://arxiv.org/abs/1312.6114)
        """
        params = self.estimate_params(input_repr)
        negative_kl_divergence = self.compute_negative_kld(params)
        z = self.reparameterize(params)
        # Apply dropout to theta.
        theta = self._z_dropout(z)

        # Normalize theta.
        theta = torch.softmax(theta, dim=-1)

        return {
                "theta": theta,
                "params": params,
                "negative_kl_divergence": negative_kl_divergence
                }

    @overrides
    def encode(self, input_vector: torch.Tensor):
        return self.encoder_d1(input_vector)

    @overrides
    def get_beta(self):
        return self._decoder._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212
