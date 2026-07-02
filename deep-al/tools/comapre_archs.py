"""
PyTorch implementation of sklearn's MLPClassifier.

Matches sklearn's MLPClassifier API as closely as possible:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import unique_labels
import warnings


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    PyTorch-based Multi-layer Perceptron classifier matching sklearn's MLPClassifier API.

    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization. Note: 'lbfgs' is mapped to Adam in PyTorch.

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.

    batch_size : int or 'auto', default='auto'
        Size of minibatches for stochastic optimizers.
        If 'auto', batch_size=min(200, n_samples).

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.

    learning_rate_init : float, default=0.001
        Initial learning rate.

    power_t : float, default=0.5
        Exponent for inverse scaling learning rate. Used when learning_rate='invscaling'.

    max_iter : int, default=200
        Maximum number of iterations (epochs).

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration.

    random_state : int or None, default=None
        Random state for reproducibility.

    tol : float, default=1e-4
        Tolerance for optimization. Training stops when loss improvement < tol
        for n_iter_no_change consecutive epochs.

    verbose : bool, default=False
        Whether to print progress messages.

    warm_start : bool, default=False
        When True, reuse solution of previous fit as initialization.

    momentum : float, default=0.9
        Momentum for gradient descent update. Used with SGD solver.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum. Used with SGD and momentum > 0.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation score
        is not improving.

    validation_fraction : float, default=0.1
        Proportion of training data to set aside for validation (when early_stopping=True).

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector (Adam only).

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector (Adam only).

    epsilon : float, default=1e-8
        Value for numerical stability in Adam optimizer.

    n_iter_no_change : int, default=10
        Maximum number of epochs with no improvement before stopping.

    max_fun : int, default=15000
        Maximum number of loss function calls (not used, kept for API compatibility).
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_activation(self):
        mapping = {
            "identity": nn.Identity(),
            "logistic": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
        }
        if self.activation not in mapping:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Choose from {list(mapping.keys())}."
            )
        return mapping[self.activation]

    def _build_network(self, n_features, n_outputs):
        layers = []
        in_size = n_features
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(self._get_activation())
            in_size = h
        layers.append(nn.Linear(in_size, n_outputs))
        return nn.Sequential(*layers)

    def _get_optimizer(self):
        params = self.network_.parameters()
        if self.solver in ("adam", "lbfgs"):
            return optim.Adam(
                params,
                lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
                weight_decay=self.alpha,
            )
        elif self.solver == "sgd":
            return optim.SGD(
                params,
                lr=self.learning_rate_init,
                momentum=self.momentum,
                nesterov=self.nesterovs_momentum and self.momentum > 0,
                weight_decay=self.alpha,
            )
        else:
            raise ValueError(
                f"Unknown solver '{self.solver}'. Choose from 'adam', 'sgd', 'lbfgs'."
            )

    def _update_learning_rate(self, optimizer, epoch):
        if self.learning_rate == "constant":
            return
        elif self.learning_rate == "invscaling":
            lr = self.learning_rate_init / (epoch + 1) ** self.power_t
        elif self.learning_rate == "adaptive":
            # Reduce by factor 10 if no improvement (handled in train loop)
            return
        else:
            raise ValueError(f"Unknown learning_rate '{self.learning_rate}'.")
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_outputs = len(self.classes_)
        self.n_outputs_ = n_outputs

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Early-stopping split
        if self.early_stopping:
            n_val = max(1, int(n_samples * self.validation_fraction))
            idx = np.random.permutation(n_samples)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            X_val, y_val = X[val_idx], y_encoded[val_idx]
            X_train, y_train = X[train_idx], y_encoded[train_idx]
        else:
            X_train, y_train = X, y_encoded

        # Build (or reuse) network
        if not self.warm_start or not hasattr(self, "network_"):
            self.network_ = self._build_network(n_features, n_outputs)

        optimizer = self._get_optimizer()

        is_binary = n_outputs == 2
        if is_binary:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Batch size
        bs = (
            min(200, len(X_train))
            if self.batch_size == "auto"
            else self.batch_size
        )

        X_t = torch.tensor(X_train)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=bs, shuffle=self.shuffle)

        self.loss_curve_ = []
        self.best_loss_ = np.inf
        self.best_validation_score_ = -np.inf
        no_improve_count = 0
        adaptive_lr_no_improve = 0

        self.n_iter_ = 0

        for epoch in range(self.max_iter):
            self.network_.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.network_(X_batch)

                if is_binary:
                    loss = criterion(
                        logits[:, 1].float(), y_batch.float()
                    )
                else:
                    loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_curve_.append(avg_loss)
            self.n_iter_ += 1

            self._update_learning_rate(optimizer, epoch)

            if self.verbose:
                print(f"Iteration {epoch + 1}, loss = {avg_loss:.6f}")

            # Early stopping / no-change check
            if self.early_stopping:
                val_score = self.score(X_val, self.label_encoder_.inverse_transform(y_val))
                if val_score - self.best_validation_score_ > self.tol:
                    self.best_validation_score_ = val_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}.")
                    break
            else:
                if self.best_loss_ - avg_loss > self.tol:
                    self.best_loss_ = avg_loss
                    no_improve_count = 0
                    adaptive_lr_no_improve = 0
                else:
                    no_improve_count += 1
                    adaptive_lr_no_improve += 1

                # Adaptive LR: reduce by factor of 10 when no improvement
                if self.learning_rate == "adaptive" and adaptive_lr_no_improve >= self.n_iter_no_change:
                    for pg in optimizer.param_groups:
                        pg["lr"] /= 10.0
                    adaptive_lr_no_improve = 0
                    if self.verbose:
                        new_lr = optimizer.param_groups[0]["lr"]
                        print(f"  Reducing learning rate to {new_lr:.2e}")

                if no_improve_count >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Converged at epoch {epoch + 1}.")
                    break
        else:
            warnings.warn(
                f"Stochastic Optimizer: Maximum iterations ({self.max_iter}) reached "
                "and the optimization hasn't converged yet.",
                stacklevel=2,
            )

        self.t_ = self.n_iter_ * (len(X_train) // bs + 1)
        return self

    # ------------------------------------------------------------------
    # predict helpers
    # ------------------------------------------------------------------

    def _forward(self, X):
        if not hasattr(self, "network_"):
            raise ValueError(
                "This MLPClassifier instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.network_.eval()
        with torch.no_grad():
            return self.network_(X)

    # ------------------------------------------------------------------
    # predict / predict_proba / predict_log_proba
    # ------------------------------------------------------------------

    def predict(self, X):
        """Predict class labels for samples in X."""
        logits = self._forward(X)
        preds = torch.argmax(logits, dim=1).numpy()
        return self.label_encoder_.inverse_transform(preds)

    def predict_proba(self, X):
        """Probability estimates."""
        logits = self._forward(X)
        if self.n_outputs_ == 2:
            probs = torch.sigmoid(logits[:, 1]).numpy()
            return np.column_stack([1 - probs, probs])
        probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict_log_proba(self, X):
        """Return the log of probability estimates."""
        return np.log(self.predict_proba(X))

    # ------------------------------------------------------------------
    # sklearn compatibility
    # ------------------------------------------------------------------

    def score(self, X, y):
        """Return mean accuracy on the given test data and labels."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def _more_tags(self):
        return {"multilabel": False}


# ---------------------------------------------------------------------------
# Quick smoke-test / usage demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier as SklearnMLP

    print("=" * 60)
    print("Iris dataset (multi-class)")
    print("=" * 60)
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # PyTorch version
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        verbose=False,
    )
    clf.fit(X_train, y_train)
    print(f"PyTorch MLP  accuracy: {clf.score(X_test, y_test):.4f}")

    # sklearn version
    sk_clf = SklearnMLP(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
    )
    sk_clf.fit(X_train, y_train)
    print(f"sklearn MLP  accuracy: {sk_clf.score(X_test, y_test):.4f}")

    print()
    print("=" * 60)
    print("Breast cancer dataset (binary)")
    print("=" * 60)
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf2 = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=300,
        random_state=42,
    )
    clf2.fit(X_train, y_train)
    print(f"PyTorch MLP  accuracy: {clf2.score(X_test, y_test):.4f}")

    sk_clf2 = SklearnMLP(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=300,
        random_state=42,
    )
    sk_clf2.fit(X_train, y_train)
    print(f"sklearn MLP  accuracy: {sk_clf2.score(X_test, y_test):.4f}")

    print()
    print("predict_proba shape:", clf2.predict_proba(X_test[:5]).shape)
    print("predict_log_proba shape:", clf2.predict_log_proba(X_test[:5]).shape)
    print("classes_:", clf2.classes_)
    print("n_iter_:", clf2.n_iter_)
    print("loss_curve_ (last 5):", clf2.loss_curve_[-5:])