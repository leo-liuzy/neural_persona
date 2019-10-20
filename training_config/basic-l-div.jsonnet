local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

local BASE_READER(LAZY) = {
  "lazy": LAZY == 1,
  "type": "entity_based_reader"
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "validation_dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER"))),
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "type": "extended_vocabulary",
      "directory_path": std.extVar("VOCABULARY_DIRECTORY")
   },
   "model": {
      "type": "basic-l-div",
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "entity_based",
         "ignore_oov": true
      },
      "doc_saturation_period": std.extVar("DOC_SATURATION_PERIOD"),
      "doc_period": std.extVar("DOC_PERIOD"),
      "doc_kl_weight_annealing": std.extVar("DOC_KL_ANNEALING"),
      "doc_sigmoid_weight_1": std.extVar("DOC_SIGMOID_WEIGHT_1"),
      "doc_sigmoid_weight_2": std.extVar("DOC_SIGMOID_WEIGHT_2"),
      "doc_linear_scaling": std.extVar("DOC_LINEAR_SCALING"),
      "entity_saturation_period": std.extVar("ENTITY_SATURATION_PERIOD"),
      "entity_period": std.extVar("ENTITY_PERIOD"),
      "entity_kl_weight_annealing": std.extVar("ENTITY_KL_ANNEALING"),
      "entity_sigmoid_weight_1": std.extVar("ENTITY_SIGMOID_WEIGHT_1"),
      "entity_sigmoid_weight_2": std.extVar("ENTITY_SIGMOID_WEIGHT_2"),
      "entity_linear_scaling": std.extVar("ENTITY_LINEAR_SCALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "update_background_freq": std.parseInt(std.extVar("UPDATE_BACKGROUND_FREQUENCY")) == 1,
      "track_npmi": std.parseInt(std.extVar("TRACK_NPMI")) == 1,
      "background_data_path": std.extVar("BACKGROUND_DATA_PATH"),
      "vae": {
         "z_dropout": std.extVar("Z_DROPOUT"),
         "apply_batchnorm_on_normal": std.parseInt(std.extVar("APPLY_BATCHNORM_ON_NORMAL")) == 1,
         "apply_batchnorm_on_decoder": std.parseInt(std.extVar("APPLY_BATCHNORM_ON_DECODER")) == 1,
         "batchnorm_weight_learnable": std.parseInt(std.extVar("BATCHNORM_WEIGHT_LEARNABLE")) == 1,
         "batchnorm_bias_learnable": std.parseInt(std.extVar("BATCHNORM_BIAS_LEARNABLE")) == 1,
         "stochastic_beta": std.parseInt(std.extVar("STOCHASTIC_BETA")) == 1,
         "encoder_topic": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("K"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection_topic": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("K"))),
            "input_dim": std.extVar("K"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
        "log_variance_projection_topic": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("K"))),
            "input_dim": std.parseInt(std.extVar("K")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },
         "encoder_entity": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 1 ,
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection_entity": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.extVar("P"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
        "log_variance_projection_entity": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },
         "encoder_entity_approx": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.extVar("K"),
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection_entity_approx": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.extVar("P"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
         "log_variance_projection_entity_approx": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },

         "decoder_persona": {
            "activations": "linear",
            "hidden_dims": [std.parseInt(std.extVar("VOCAB_SIZE")) + 1 ],
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": 1
         },
         "decoder_topic": {
            "activations": "linear",
            "hidden_dims": [std.parseInt(std.extVar("P"))],
            "input_dim": std.parseInt(std.extVar("K")),
            "num_layers": 1
         },
         "decoder_mean_projection_topic": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
         "decoder_log_variance_projection_topic": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },
         "type": "basic-l"
      }
   },
    "iterator": {
      "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
      "track_epoch": true,
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_serialized_models_to_keep": 1,
      "num_epochs": 100,
      "patience": 10,
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam"
      },
      "validation_metric": std.extVar("VALIDATION_METRIC")
   }
}
