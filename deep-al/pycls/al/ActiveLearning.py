# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg, train_labels=None, lset=None):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        self.train_labels = train_labels  # Store train_labels for methods that need it
        self.sampling_fn = self.choose_sampling_function(train_labels, lset)
        
        # CK manager will be initialized externally in train_al.py if needed
        self.ck_manager = None

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None, 
                         per_class_accuracy=None, data_obj=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.
        
        per_class_accuracy: Per-class accuracy vector for NN fusion mode (optional)
        
        data_obj: Data object for creating data loaders (optional, required for NN fusion)

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if self.sampling_fn is not None and hasattr(self.sampling_fn, 'select_samples'):
            logger.info(f"Using {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} sampling function for active learning.")
            # For BAYES_MISP with NN fusion, pass additional parameters
            nn_fusion_enabled = getattr(self.cfg, 'NN_FUSION', False)
            if nn_fusion_enabled and per_class_accuracy is not None and clf_model is not None:
                # Set NN fusion parameters before sampling
                if hasattr(self.sampling_fn, 'set_nn_fusion_params'):
                    self.sampling_fn.set_nn_fusion_params(
                        clf_model=clf_model,
                        data_obj=data_obj,
                        train_data=trainDataset,
                        per_class_accuracy=per_class_accuracy
                    )
            return self.sampling_fn.select_samples(lSet, uSet)


        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random_1c":

            activeSet, uSet = self.sampler.random_1c(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                                                      dataset=trainDataset, train_labels=self.train_labels)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            # Handle sklearn models differently (no training/eval modes)
            if self.cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn':
                activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                    ,model=clf_model,dataset=trainDataset)
            else:
                oldmode = clf_model.training
                clf_model.eval()
                activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                    ,model=clf_model,dataset=trainDataset)
                clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            # Handle sklearn models differently (no training/eval modes)
            if self.cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn':
                activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                    ,model=clf_model,dataset=trainDataset)
            else:
                oldmode = clf_model.training
                clf_model.eval()
                activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                    ,model=clf_model,dataset=trainDataset)
                clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            # Handle sklearn models differently (no training/eval modes)
            if self.cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn':
                activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                    ,model=clf_model,dataset=trainDataset)
            else:
                oldmode = clf_model.training
                clf_model.eval()
                activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                    ,model=clf_model,dataset=trainDataset)
                clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     clf_model.cuda(0)
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = probcov.select_samples()
            # probcov.plot_tsne()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["maxherding", "max_herding"]:
            from .maxherding import MaxHerding
            delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            maxherding = MaxHerding(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, delta=delta)
            activeSet, uSet = maxherding.select_samples()
            # maxherding.plot_tsne()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            from .DCoM import DCoM
            dcom = DCoM(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                        lSet_deltas=self.cfg.ACTIVE_LEARNING.DELTA_LST)
            activeSet, uSet = dcom.select_samples(clf_model, trainDataset, self.dataObj)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, \
                uSet=uSet, clf_model=clf_model,dataset=trainDataset)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_model=clf_model, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(cfg=self.cfg, dataObj=self.dataObj)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, \
                                unlabeled_dataloader=uSet_loader, uSet=uSet)
        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
    
    def _has_ck_matrices(self):
        """Check if sampling_fn already has C_general and K_general."""
        return (self.sampling_fn is not None and 
                hasattr(self.sampling_fn, 'C_general') and 
                hasattr(self.sampling_fn, 'K_general'))
    
    def attach_ck_manager(self, ck_manager):
        """
        Attach an externally created CK manager to this ActiveLearning object.
        
        Args:
            ck_manager: CKMatrixManager instance
        """
        self.ck_manager = ck_manager
        
        # If method doesn't have its own C_general/K_general, expose CK manager's matrices
        if not self._has_ck_matrices():
            logger.info("Method doesn't have native C/K matrices - exposing CK manager matrices to sampling_fn")
            # Expose C_general and K_general as if native to sampling_fn
            if self.sampling_fn is None:
                # Create dummy object to hold matrices
                self.sampling_fn = type('CKMatrixHolder', (object,), {
                    'C_general': self.ck_manager.C_general,
                    'K_general': self.ck_manager.K_general
                })()
            else:
                # Attach to existing sampling_fn
                self.sampling_fn.C_general = self.ck_manager.C_general
                self.sampling_fn.K_general = self.ck_manager.K_general
        else:
            logger.info(f"Method {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} has native C/K matrices - CK manager is separate")

    def update_sampling_function(self, train_labels=None, lset=None):
        """
        Update the sampling function when switching AL methods.
        
        Args:
            train_labels: Labels of training data
            lset: Current labeled set indices
        """
        logger.info(f"Updating sampling function to: {self.cfg.ACTIVE_LEARNING.SAMPLING_FN}")
        
        # Store old C/K matrices if they exist
        old_C_general = None
        old_K_general = None
        if hasattr(self, 'sampling_fn') and self.sampling_fn is not None:
            if hasattr(self.sampling_fn, 'C_general'):
                old_C_general = self.sampling_fn.C_general
            if hasattr(self.sampling_fn, 'K_general'):
                old_K_general = self.sampling_fn.K_general
        
        # Create new sampling function
        self.sampling_fn = self.choose_sampling_function(train_labels, lset)
        
        # Check if we need to preserve C/K matrices for pseudo-labeling or distillation
        needs_ck_matrices = (
            self.cfg.TRAIN_PSEUDO_LABELS or 
            getattr(self.cfg, 'DISTILLATION_TRAINING', False)
        )
        
        # If ck_manager exists, update its reference in the new sampling_fn
        if self.ck_manager is not None:
            if self.sampling_fn is None:
                # Only create holder if we actually need C/K matrices
                if needs_ck_matrices:
                    self.sampling_fn = type('CKMatrixHolder', (object,), {
                        'C_general': self.ck_manager.C_general,
                        'K_general': self.ck_manager.K_general
                    })()
            else:
                # Attach to new sampling_fn only if it doesn't have its own matrices
                # (methods like bayes_misp build their own K/C matrices in specific formats)
                if not hasattr(self.sampling_fn, 'C_general'):
                    self.sampling_fn.C_general = self.ck_manager.C_general
                if not hasattr(self.sampling_fn, 'K_general'):
                    self.sampling_fn.K_general = self.ck_manager.K_general
        # Otherwise, restore old C/K matrices if needed and they existed
        elif old_C_general is not None and not self._has_ck_matrices() and needs_ck_matrices:
            if self.sampling_fn is None:
                # Only create holder if we actually need C/K matrices
                self.sampling_fn = type('CKMatrixHolder', (object,), {
                    'C_general': old_C_general,
                    'K_general': old_K_general
                })()
            else:
                # Attach to new sampling_fn if it doesn't have them
                if not hasattr(self.sampling_fn, 'C_general'):
                    self.sampling_fn.C_general = old_C_general
                if not hasattr(self.sampling_fn, 'K_general') and old_K_general is not None:
                    self.sampling_fn.K_general = old_K_general
        
        logger.info(f"Sampling function updated successfully")

    def choose_sampling_function(self, train_labels=None, lset=None):
        # if self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
        #     from .prob_cover import ProbCover
        #     probcov = ProbCover(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
        #                     delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
        #     return probcov
        #     activeSet, uSet = probcov.select_samples(lSet, uSet)
            # probcov.plot_tsne()

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover_matrix", "max_herding_matrix"]:
            from .coverage_matrix_methods import CoverageMatrixMethod
            cmm = CoverageMatrixMethod(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        train_labels= train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
            return cmm

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover_vector", "max_herding_vector"]:
            from .coverage_vector_methods import CoverageVectorMethod
            cvm = CoverageVectorMethod(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        train_labels= train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
            return cvm

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["bayes_misp"]:
            from .BAYES_MISP import BAYES_MISP

            bayes_misp = BAYES_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        train_labels= train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
            return bayes_misp

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["bayes_misp_v1"]:
            from .BAYES_MISP_v1 import BAYES_MISP

            bayes_misp = BAYES_MISP(self.cfg, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        train_labels= train_labels, delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA, lset=lset)
            return bayes_misp
