local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));
local EXTRACTER_OUTPUT_DIM = std.length(std.parseJson(std.extVar("NGRAM_FILTER_SIZES"))) * std.parseInt(std.extVar("NUM_FILTER"));
local BASE_READER(LAZY, USE_DOC_INFO) = {
  "lazy": LAZY == 1,
  "type": "ladder_reader",
  "use_doc_info": USE_DOC_INFO == 1
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), std.parseInt(std.extVar("USE_DOC_INFO"))),
   "validation_dataset_reader": BASE_READER(std.parseInt(std.extVar("LAZY_DATASET_READER")), std.parseInt(std.extVar("USE_DOC_INFO"))),
   "train_data_path": std.extVar("TRAIN_PATH"),
   "validation_data_path": std.extVar("DEV_PATH"),
   "vocabulary": {
      "type": "extended_vocabulary",
      "directory_path": std.extVar("VOCABULARY_DIRECTORY")
   },
   "model": {
      "type": "ladder",
      "bow_embedder": {
         "type": "bag_of_word_counts",
         "vocab_namespace": "ladder",
         "ignore_oov": true
      },
      "apply_batchnorm_on_recon": std.parseInt(std.extVar("APPLY_BATCHNORM_ON_RECON")) == 1,
      "batchnorm_weight_learnable": std.parseInt(std.extVar("BATCHNORM_WEIGHT_LEARNABLE")) == 1,
      "batchnorm_bias_learnable": std.parseInt(std.extVar("BATCHNORM_BIAS_LEARNABLE")) == 1,
      "kl_weight_annealing": std.extVar("KL_ANNEALING"),
      "sigmoid_weight_1": std.extVar("SIGMOID_WEIGHT_1"),
      "sigmoid_weight_2": std.extVar("SIGMOID_WEIGHT_2"),
      "linear_scaling": std.extVar("LINEAR_SCALING"),
      "reference_counts": std.extVar("REFERENCE_COUNTS"),
      "reference_vocabulary": std.extVar("REFERENCE_VOCAB"),
      "use_doc_info": std.parseInt(std.extVar("USE_DOC_INFO")) == 1,
      "use_background": std.parseInt(std.extVar("USE_BACKGROUND")) == 1,
      "background_data_path": std.extVar("BACKGROUND_DATA_PATH"),
      "update_background_freq": std.parseInt(std.extVar("UPDATE_BACKGROUND_FREQUENCY")) == 1,
      "track_npmi": std.parseInt(std.extVar("TRACK_NPMI")) == 1,
      "vae": {
         "z_dropout": std.extVar("Z_DROPOUT"),
         "prior": std.parseJson(std.extVar("PRIOR")),
         "apply_batchnorm_on_normal": std.parseInt(std.extVar("APPLY_BATCHNORM_ON_NORMAL")) == 1,
         "apply_batchnorm_on_decoder": std.parseInt(std.extVar("APPLY_BATCHNORM_ON_DECODER")) == 1,
         "batchnorm_weight_learnable": std.parseInt(std.extVar("BATCHNORM_WEIGHT_LEARNABLE")) == 1,
         "batchnorm_bias_learnable": std.parseInt(std.extVar("BATCHNORM_BIAS_LEARNABLE")) == 1,
         "stochastic_beta": std.parseInt(std.extVar("STOCHASTIC_BETA")) == 1,
         "extracter": {
            "type": "cnn",
            "embedding_dim": std.parseInt(std.extVar("VOCAB_SIZE")) + 1,
            "num_filters": std.parseInt(std.extVar("NUM_FILTER")),
            "ngram_filter_sizes": std.parseJson(std.extVar("NGRAM_FILTER_SIZES"))
         },
         "encoder_d1": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": EXTRACTER_OUTPUT_DIM,
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection_d1": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.extVar("P"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
         "log_variance_projection_d1": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },

         "encoder_d2": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("K"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection_d2": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("K"))),
            "input_dim": std.extVar("K"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
         "log_variance_projection_d2": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("K"))),
            "input_dim": std.parseInt(std.extVar("K")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },

         "encoder_t1": {
            "activations": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.extVar("ENCODER_ACTIVATION")),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("K")),
            "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS"))
         },
         "mean_projection_t1": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.extVar("P"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
         "log_variance_projection_t1": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },

         "decoder1": {
            "activations": "linear",
            "hidden_dims": [EXTRACTER_OUTPUT_DIM],
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": 1
         },
         "decoder2": {
            "activations": "linear",
            "hidden_dims": [std.parseInt(std.extVar("P"))],
            "input_dim": std.parseInt(std.extVar("K")),
            "num_layers": 1
         },
         "mean_projection_dec2": {
            "activations": std.extVar("MEAN_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.extVar("P"),
            "num_layers": std.parseInt(std.extVar("NUM_MEAN_PROJECTION_LAYERS"))
         },
         "log_variance_projection_dec2": {
            "activations": std.extVar("LOG_VAR_PROJECTION_ACTIVATION"),
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS")), function(i) std.parseInt(std.extVar("P"))),
            "input_dim": std.parseInt(std.extVar("P")),
            "num_layers": std.parseInt(std.extVar("NUM_LOG_VAR_PROJECTION_LAYERS"))
         },
 "type": "ladder"
      }
   },
    "iterator": {
      "batch_size": std.parseInt(std.extVar("BATCH_SIZE")),
      "track_epoch": true,
      "type": "basic",
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_serialized_models_to_keep": 1,
      "num_epochs": 50,
      "patience": 5,
      "optimizer": {
         "lr": std.extVar("LEARNING_RATE"),
         "type": "adam"
      },
      "validation_metric": std.extVar("VALIDATION_METRIC")
   }
}
