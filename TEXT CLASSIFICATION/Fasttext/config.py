from tensorflow import flags
# Model Hyperparameters
flags.DEFINE_integer("embedding_dim", 300,
                     "Dimensionality of character embedding (default: 128)")
flags.DEFINE_integer("hidden_num", 128,
                     "Number of hidden layer (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5,
                   "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 1,
                   "L2 regularizaion lambda (default: 0.0)")
flags.DEFINE_float("learning_rate", 0.0001, "learn rate( default: 0.0)")
flags.DEFINE_integer('extend_feature_dim', 10, 'overlap_feature_dim')
# Training parameters
flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
flags.DEFINE_integer("n_fold", 10, "the number of Cross-validation for mr,subj,cr,mpqa")
flags.DEFINE_boolean(
    "trainable", True, "is embedding trainable? (default: False)")
flags.DEFINE_integer(
    "num_epochs", 20, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 500,
                     "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500,
                     "Save model after this many steps (default: 100)")
flags.DEFINE_boolean('dns', 'False', 'whether use dns or not')
flags.DEFINE_string('data', 'TREC', 'data set: mr;subj;cr;mpqa;sst2;TREC')
# flags.DEFINE_string('model_name', 'PE_reduce', 'select the model you need')
flags.DEFINE_boolean("allow_soft_placement", True,
                     "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False,
                     "Log placement of ops on devices")
flags.DEFINE_boolean('isEnglish', True, 'whether data is english')
