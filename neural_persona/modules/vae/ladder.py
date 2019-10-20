from typing import Dict
import os
import torch
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from ipdb import set_trace as bp
from overrides import overrides

from neural_persona.common.util import normal_kl, create_trainable_BatchNorm1d, EPSILON
from allennlp.models import Model
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
                 extracter: Seq2VecEncoder,
                 encoder_d1: FeedForward,
                 encoder_d2: FeedForward,
                 mean_projection_d1: FeedForward,
                 log_variance_projection_d1: FeedForward,
                 mean_projection_d2: FeedForward,
                 log_variance_projection_d2: FeedForward,
                 encoder_t1: FeedForward,
                 mean_projection_t1: FeedForward,
                 log_variance_projection_t1: FeedForward,
                 decoder1: FeedForward,
                 decoder2: FeedForward,
                 mean_projection_dec2: FeedForward,
                 log_variance_projection_dec2: FeedForward,
                 prior: Dict = {"type": "normal", "mu": 0, "var": 1},
                 apply_batchnorm_on_normal: bool = False,
                 apply_batchnorm_on_decoder: bool = False,
                 batchnorm_weight_learnable: bool = False,
                 batchnorm_bias_learnable: bool = True,
                 stochastic_beta: bool = False,
                 z_dropout: float = 0.2) -> None:
        super(LadderVAE, self).__init__(vocab)
        self.extracter = extracter
        # bp() 
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
        self.mean_projection_dec2 = mean_projection_dec2
        self.log_variance_projection_dec2 = log_variance_projection_dec2

        self._z_dropout = torch.nn.Dropout(z_dropout)

        self.num_persona = mean_projection_d1.get_output_dim()
        self.num_topic = mean_projection_d2.get_output_dim()

        self.prior = prior
        self.p_params = None
        # self.p_mu, self.p_sigma, self.p_log_var = None, None, None
        self.initialize_prior(prior)

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
            self.mean_bn_t1 = create_trainable_BatchNorm1d(self.num_persona,
                                                           weight_learnable=batchnorm_weight_learnable,
                                                           bias_learnable=batchnorm_bias_learnable,
                                                           eps=0.001, momentum=0.001, affine=True)
            self.log_var_bn_t1 = create_trainable_BatchNorm1d(self.num_persona,
                                                              weight_learnable=batchnorm_weight_learnable,
                                                              bias_learnable=batchnorm_bias_learnable,
                                                              eps=0.001, momentum=0.001, affine=True)
            self.mean_bn_dec2 = create_trainable_BatchNorm1d(self.num_persona,
                                                             weight_learnable=batchnorm_weight_learnable,
                                                             bias_learnable=batchnorm_bias_learnable,
                                                             eps=0.001, momentum=0.001, affine=True)
            self.log_var_bn_dec2 = create_trainable_BatchNorm1d(self.num_persona,
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
    def forward(self, input_vector: torch.FloatTensor):  # pylint: disable = W0221
        """
        Given the input representation, produces the reconstruction from theta
        as well as the negative KL-divergence, theta itself, and the parameters
        of the distribution.
        """
        output = {}
        _, max_num_p, x_dim = input_vector.shape
        mask = input_vector.sum(dim=-1) != 0
        # mask = tmp.repeat(1, 1, x_dim)
        # bp()
        input_vector = self.extracter(input_vector, mask)
        # bp()
        # bottom-up inference -- q(z_2 | x)
        input_repr1 = self.encoder_d1(input_vector)
        d_1 = self.estimate_params(input_repr1, self.mean_projection_d1, self.log_variance_projection_d1,
                                   self.mean_bn_d1, self.log_var_bn_d1)
        input_repr2 = self.encoder_d2(d_1['mean'])
        d_2 = self.estimate_params(input_repr2, self.mean_projection_d2, self.log_variance_projection_d2,
                                   self.mean_bn_d2, self.log_var_bn_d2)
        output["negative_kl_divergence"] = self.compute_negative_kld(d_2, {"mean": self.p_mu, "log_variance": self.p_log_var})

        # sample an inferred z_2
        z_2 = self.reparameterize(params=d_2)
        z_2 = self._z_dropout(z_2)

        # get topic representation
        theta_t = torch.softmax(z_2, dim=-1)
        output["theta_t"] = theta_t

        # top-down inference -- q(z_1 | z_2, x) TODO: should I use theta_t or z_1 to infer t_1?
        t_1 = self.estimate_params(z_2, self.mean_projection_t1, self.log_variance_projection_t1,
                                   self.mean_bn_t1, self.log_var_bn_t1)
        qz_1_params = self.merge_normal(param_d=d_1, param_t=t_1)

        # decode topic representation to input to calculate persona representation
        beta_t = self._decoder2.weight.t()
        if self._apply_batchnorm_on_decoder:
            beta_t = self.decoder_bn2(beta_t)
        if self._stochastic_beta:
            beta_t = torch.nn.functional.softmax(beta_t, dim=1)
        h_1 = theta_t @ beta_t
        pz_1_params = self.estimate_params(h_1, self.mean_projection_dec2, self.log_variance_projection_dec2,
                                           self.mean_bn_dec2, self.log_var_bn_dec2)

        output["negative_kl_divergence"] += self.compute_negative_kld(qz_1_params, pz_1_params)

        # turn inferred z_1 to persona representation
        # z_1 = self.reparameterize(qz_1_params)
        z_1 = qz_1_params["mean"]
        z_1 = self._z_dropout(z_1)
        theta_p = torch.softmax(z_1, dim=-1)
        output["theta_p"] = theta_p

        # decode persona representation to word reconstruction
        beta_p = self._decoder1.weight.t()
        if self._apply_batchnorm_on_decoder:
            beta_p = self.decoder_bn1(beta_p)
        if self._stochastic_beta:
            beta_p = torch.nn.functional.softmax(beta_p, dim=1)
        reconstruction = theta_p @ beta_p
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
        log_var = torch.log(var + EPSILON)

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
    def compute_negative_kld(self, q_params: Dict, p_params: Dict):
        """
        Compute the closed-form solution for negative KL-divergence for Gaussians.
        """
        mu, log_var = q_params["mean"], q_params["log_variance"]  # pylint: disable=C0103
        # bp()
        negative_kl_divergence = -normal_kl((mu, log_var), (p_params["mean"], p_params["log_variance"]))
        # bp()
        return negative_kl_divergence

    def reparameterize(self, params):
        """z is the result of the reparameterization trick."""
        mu, sigma = params["mean"], params["sigma"]  # pylint: disable=C0103

        # Generate random noise and sample theta.
        # Shape: (batch, latent_dim)
        batch_size = params["mean"].size(0)
        latent_dim = params["mean"].size(-1)

        # Enable reparameterization for training only.
        if self.training:
            seed = os.environ['SEED']
            torch.manual_seed(seed)
            # Seed all GPUs with the same seed if available.
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            epsilon = torch.randn(batch_size, latent_dim).to(device=mu.device)
            z = mu + sigma * epsilon  # pylint: disable=C0103
        else:
            z = mu  # pylint: disable=C0103
        return z

    @overrides
    def get_beta(self, level: str = "p"):
        if level == "p":
            return self._decoder1._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212
        elif level == "t":
            return self._decoder2._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212
        else:
            ValueError()
