from pycls.core.config import cfg
from tools.train_al import define_eval_model_type


def apply_train_al_args(args):
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.initial_delta
    cfg.ACTIVE_LEARNING.INITIAL_SIGMA = args.initial_sigma
    cfg.CK_SIGMA = args.ck_sigma if args.ck_sigma is not None else args.initial_sigma
    cfg.KERNEL_TYPE = args.kernel_type
    cfg.DIFF_METHOD = args.diff_method
    cfg.CONT_METHOD = args.cont_method
    cfg.DISTRIBUTION_CONT_WEIGHT_METHOD = args.distribution_cont_weight_method
    cfg.C_NORMALIZATION = args.c_normalization
    cfg.CLASS_WEIGHTING_METHOD = args.class_weighting_method
    cfg.SPARSE_K = args.sparse_K
    cfg.K_SPARSITY_THRESHOLD = args.K_sparsity_threshold if cfg.KERNEL_TYPE != "tophat" else cfg.ACTIVE_LEARNING.INITIAL_DELTA
    cfg.PSEUDO_LABELS_THRESHOLD = args.pseudo_labels_threshold
    cfg.TRAIN_PSEUDO_LABELS = args.train_pseudo_labels
    cfg.ALPHA_LOWER_BOUND = args.alpha_lower_bound
    cfg.ALPHA_UPPER_BOUND = args.alpha_upper_bound
    cfg.LOCAL_ALPHA = args.local_alpha
    cfg.LOCAL_ALPHA_ORACLE_METHOD = args.local_alpha_oracle_method
    cfg.USE_K_TOP50_MASK = args.use_k_top50_mask
    cfg.UPDATE_K_MATRIX = args.update_k_matrix
    cfg.UPDATE_K_MATRIX_FACTOR = args.update_k_matrix_factor
    cfg.UPDATE_C_MATRIX = args.update_c_matrix
    cfg.DECREASING_ALPHA = args.decrease_alpha
    debug = cfg.DEBUG = args.debug
    cfg.HIGH_BUDGET = args.high_budget
    cfg.CONFIDENCE_METHOD = args.confidence_method
    cfg.RNG_SEED = 0 if debug else args.seed
    cfg.ALPHA = args.alpha
    cfg.CK_ALPHA = args.ck_alpha if args.ck_alpha is not None else args.alpha
    cfg.CK_C_NORMALIZATION = args.ck_c_normalization
    cfg.CK_K_SPARSITY_THRESHOLD = args.ck_K_sparsity_threshold
    cfg.CK_SPARSE_K = args.ck_sparse_K
    cfg.NORM_IMPORTANCE = args.norm_importance
    cfg.SPARSE_DS = args.sparse_ds
    cfg.PSEUDO_LABEL_WEIGHT = args.pseudo_label_weight
    cfg.PSEUDO_LABEL_WEIGHTING_FUNC = args.pseudo_label_weighting_func
    cfg.PSEUDO_LABEL_CLASS_THRESHOLD = args.pseudo_label_class_threshold
    cfg.PSEUDO_LABEL_CLASS_THRESHOLD_TOPK = args.pseudo_label_class_threshold_topk
    cfg.SWITCH_ALPHA_LOW_TO_HIGH = args.switch_alpha_low_to_high
    cfg.SWITCH_ALPHA_HIGH_TO_LOW = args.switch_alpha_high_to_low
    cfg.SWITCH_ALPHA_ALLTIME = args.switch_alpha_alltime
    cfg.ALPHA_INIT_MODE = args.alpha_init_mode
    cfg.ALPHA_VECTOR_PATH = args.alpha_vector_path
    cfg.NN_FUSION = args.nn_fusion
    cfg.NN_FUSION_START_ROUND = args.nn_fusion_start_round
    cfg.NN_FUSION_INTERVAL = args.nn_fusion_interval
    cfg.DISTILLATION_TRAINING = args.distillation_training
    cfg.DISTILLATION_THRESHOLD = args.distillation_threshold
    cfg.DISTILLATION_TEMPERATURE = args.distillation_temperature
    cfg.DISTILL_FACTOR = args.distill_factor
    cfg.DISTILL_SOFT_TARGET_NORMALIZATION = args.distill_soft_target_normalization
    cfg.ALPHA_DECAY_GAMMA = args.alpha_decay_gamma
    cfg.CALC_METHOD = args.calc_method
    cfg.SWITCH_METHOD_AT_ROUND = args.switch_method_at_round
    cfg.SWITCH_TO_METHOD = args.switch_to_method
    cfg.ACTIVE_LEARNING.A_LOGISTIC = args.a_logistic
    cfg.ACTIVE_LEARNING.K_LOGISTIC = args.k_logistic
    cfg.EVAL_MODEL_TYPE = args.eval_model_type
    cfg.ACTIVE_LEARNING.MAX_ITER = args.max_iter
    cfg.ACTIVE_LEARNING.START_ITER = args.start_iter
    cfg.ACTIVE_LEARNING.EVAL_FREQUENCY = args.eval_frequency
    cfg.FIXED_DISTILL_MLP = args.fixed_distill_mlp
    define_eval_model_type(cfg, debug)

    return cfg
