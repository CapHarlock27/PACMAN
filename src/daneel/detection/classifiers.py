import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import load_model


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    precision_recall_curve,
    average_precision_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

from daneel.detection.data_process import load_data


class RandomForestClassifier:
    def __init__(self, params: dict):

        # Read -related parameters from "params.yaml"
        rf_params = params.get("rf", {})

        self.csv_path = rf_params.get("dataset_path", "tess_data.csv")
        self.n_bins = int(rf_params.get("n_bins", 1000))
        self.samples_per_class = int(rf_params.get("samples_per_class", 350))
        self.n_estimators = int(rf_params.get("n_estimators", 1000))
        self.max_depth = int(rf_params.get("max_depth", 20))
        self.min_samples_leaf = int(rf_params.get("min_samples_leaf", 1)) 
        self.max_features = (rf_params.get("max_features", 'sqrt'))
        self.bootstrap = (rf_params.get("bootstrap", True))
        self.n_jobs = (rf_params.get("n_jobs", -1))
        self.oob_score = (rf_params.get("oob_score", True))
        self.results_dir = rf_params.get("results_dir", "./rf_results")

        if self.results_dir is None or not os.path.exists(self.results_dir):
            print("RF results_dir not specified or does not exist, please input a valid path in params.yaml")
            exit()
            

        self.saved_model_path = os.path.join(self.results_dir, "tess_rf_model.joblib")
        self.saved_threshold_path = os.path.join(self.results_dir, "rf_optimal_threshold.npy")

        os.makedirs(self.results_dir, exist_ok=True)
        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
    
    def artifacts_exist(self):
        """Check if pretrained model, scaler, and threshold already exist."""
        return (
            os.path.exists(self.saved_model_path) and
            os.path.exists(self.saved_threshold_path)
        )
    
    #BALANCED AUGMENTATION
    def create_balanced_dataset(self, X, y, samples_per_class=400):
        print("\n" + "="*70)
        print("CREATING BALANCED DATASET")
        print("="*70)
        
        X0 = X[y == 0]
        X1 = X[y == 1]
        print(f"Original - Class 0: {len(X0)}, Class 1: {len(X1)}")
    
        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]
            
            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
                aug_type = np.random.rand()
                if aug_type < 0.25:
                    X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    shifts = np.random.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)
                X_result.append(X_aug)
            X_final = np.vstack(X_result)
            return X_final[:n_target]
        
        X0_bal = augment_to_target(X0, samples_per_class)
        X1_bal = augment_to_target(X1, samples_per_class)
        print(f"Balanced - Class 0: {len(X0_bal)}, Class 1: {len(X1_bal)}")
        
        X_bal = np.vstack([X0_bal, X1_bal])
        y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
            
        idx = np.arange(len(X_bal))
        np.random.shuffle(idx)
        return X_bal[idx], y_bal[idx]
    
    #DATA LOADING & PREPROCESSING
    def load_data(self, csv_path, n_bins, samples_per_class, scaler=False):
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        df = pd.read_csv(csv_path)
        print(f"Dataset: {df.shape[0]} samples")
        
        flux_cols = [f'flux_{i:04d}' for i in range(n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values
        
        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]
        
        print("Original distribution:")
        print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        if (y==0).sum() > 0:
            print(f"  Ratio: {(y==1).sum() / (y==0).sum():.2f}:1")
        
        X_train, X_test, y_train, y_test, X_err_train, X_err_test, idx_train, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"Initial split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Balance training set
        X_train, y_train = self.create_balanced_dataset(X_train, y_train, samples_per_class=samples_per_class)
        
        scaler = None
        if scaler:
            print("\n" + "="*70)
            print("STANDARDIZATION")
            print("="*70)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print(f"Train: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
            print(f"Test:  mean={X_test.mean():.6f}, std={X_test.std():.6f}")
        
        metadata_test = metadata.iloc[idx_test].reset_index(drop=True)
        print(f"Final - X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Train dist: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")
        
        # Return X_test copy (for optional inverse transform plotting if scaler is used)
        return X_train, X_test, y_train, y_test, metadata_test, X_test.copy(), X_err_test, scaler
   
   #BUILD RF MODEL
    def build_random_forest(self, X_train, y_train):

        param_grid = {
            "n_estimators": [200, 400, 600, 800, 1000],
            "max_depth": [5, 10, 20, 30, None],
            "max_features": ["auto", "sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample", None]
    }

        rf = RandomForestClassifier(random_state=42)

        search = GridSearchCV(
            rf,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=2,
        )

        search.fit(X_train, y_train)

        best_rf = search.best_estimator_
        return best_rf

    #TRAIN MODEL
    def train_model(model, X_train, y_train):
        model.fit(X_train, y_train)
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            print(f"OOB Score: {model.oob_score_:.4f}")
        return model

    #THRESHOLD OPTIMIZATION & EVALUATION
    def evaluate_with_optimal_threshold(self, model, X_test, y_test):
        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("="*70)
        
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        
        y_pred_default = (proba >= 0.5).astype(int)
        y_pred_best = (proba >= best_thresh).astype(int)
        
        auc = roc_auc_score(y_test, proba)
        acc_default = accuracy_score(y_test, y_pred_default)
        acc_best = accuracy_score(y_test, y_pred_best)
        
        print(f"Optimal threshold: {best_thresh:.4f} (default=0.5)")
        print(f"  At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"Accuracy @{best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)")
        
        print("\nClassification report (optimal threshold):")
        print(classification_report(y_test, y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0))
        
        return y_pred_best, proba, best_thresh, (fpr, tpr, thresholds)
    
   #PLOTTING FUNCTIONS
    def plot_confusion_matrix_image(self, y_true, y_pred, threshold, save_path='confusion_matrix_rf.png'):
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            xlabel='Predicted', ylabel='True',
            title=f'Confusion Matrix (threshold={threshold:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_roc_curve(self, y_true, proba, save_path='roc_curve_rf.png'):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0,1], [0,1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC={auc:.4f})')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_pr_curve(self, y_true, proba, save_path='pr_curve_rf.png'):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (AP={ap:.4f})')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_probability_histograms(self, y_true, proba, save_path='probability_hist_RF.png'):
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.hist(proba[y_true==0], bins=30, alpha=0.6, label='Non-Planet', density=True)
        ax.hist(proba[y_true==1], bins=30, alpha=0.6, label='Planet', density=True)
        ax.set_xlabel('Predicted Probability (class=1)')
        ax.set_ylabel('Density')
        ax.set_title('Predicted Probability Distributions by Class')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)


    def plot_lightcurve_sample(self, idx, X_test_standardized, X_err_test, metadata_test, scaler=None, proba=None, y_true=None, y_pred=None, save_prefix='sample_lightcurve_RF'):
        # Make a single-figure plot for one sample (no subplots)
        x_std = X_test_standardized[idx].reshape(1, -1)
        if scaler is not None:
            x_orig = scaler.inverse_transform(x_std).flatten()
            yerr = X_err_test[idx]
        else:
            x_orig = x_std.flatten()
            yerr = X_err_test[idx]
        
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.errorbar(np.arange(len(x_orig)), x_orig, yerr=yerr, fmt='o', markersize=2, alpha=0.6)
        ax.axhline(np.median(x_orig), linestyle='--', linewidth=1)
        ax.set_xlabel('Time Bin')
        ax.set_ylabel('Flux')
        
        toi = metadata_test.loc[idx, 'toi_name']
        tic = metadata_test.loc[idx, 'tic']
        disp = metadata_test.loc[idx, 'disp']
        sector = metadata_test.loc[idx, 'sector']
        
        tstr = f'TOI {toi} (TIC {tic}, {disp}) - Sector {sector}'
        if proba is not None and y_true is not None and y_pred is not None:
            pred_str = 'Transit' if y_pred[idx]==1 else 'Non-Transit'
            true_str = 'Transit' if y_true[idx]==1 else 'Non-Transit'
            tstr += f'\nTrue: {true_str} | Pred: {pred_str} (p={proba[idx]:.3f})'
        
        ax.set_title(tstr)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = f"{save_prefix}_{idx}.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
        plt.close(fig)

    def run(self):
        """
        If pretrained artifacts exist:
             load them and classify
        Else:
             train RF, save artifacts, classify
        """

        # TRAIN-ONCE 
        if self.artifacts_exist():
            print("\n=== Pretrained model found — running inference ===\n")
            return self.run_inference()

        print("\n=== No pretrained RF found — training new model ===\n")
        
        """
        Full pipeline:
        - load data from CSV
        - build & train RF
        - find optimal threshold
        - plot statistics
        - write report_rf_assignment2_taskD.txt
        """

        print("\n" + "=" * 70)
        print("LOADING & PREPARING DATA")
        print("=" * 70)

        X_train, X_test, y_train, y_test, metadata_test, X_test_orig, X_err_test, scaler = self.load_data(
            csv_path=self.csv_path, n_bins=self.n_bins, samples_per_class=self.samples_per_class, use_scaler=False
        )

        # Build model
        model = self.build_random_forest()

        # Train
        history = self.train_model(model, X_train, y_train, X_test, y_test)

        # Evaluate with optimal threshold
        y_pred_opt, proba_test, best_thresh, roc_tuple = self.evaluate_with_optimal_threshold(model, X_test, y_test)
        

        # Plot stats (confusion matrix + history + sample lightcurves)
        self.plot_confusion_matrix_image(y_test, y_pred_opt, best_thresh, save_path='confusion_matrix_rf.png')
        self.plot_roc_curve(y_test, proba_test, save_path='roc_curve_rf.png')
        self.plot_pr_curve(y_test, proba_test, save_path='pr_curve_rf.png')
        self.plot_probability_histograms(y_test, proba_test, save_path='probability_hist_RF.png')
        # Write text report (confusion matrix + precision)
        rf = confusion_matrix(y_test, y_pred_opt)
        precision = precision_recall_curve(y_test, y_pred_opt)

        report_path = os.path.join(
            self.results_dir, "report_rf_assignment2_taskD.txt"
        )
        with open(report_path, "w") as f:
            f.write("RF DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(
                classification_report(
                    y_test,
                    y_pred_opt,
                    target_names=["Non-Planet", "Planet"],
                    digits=4,
                    zero_division=0,
                )
            )
            f.write("\n\nConfusion matrix:\n")
            f.write(str(rf))
            f.write(
                f"\n\nPrecision (optimal threshold={best_thresh:.4f}): {precision:.4f}\n"
            )

        print(f"\nSaved text report to {report_path}")

        # SAVE MODEL, SCALER, THRESHOLD FOR FUTURE USE
        print("\nSaving model, scaler and threshold for future runs...")
        joblib.dump(self.saved_model_path)
        np.save(self.saved_threshold_path, best_thresh)

        print(f"Saved model: {self.saved_model_path}")
        print(f"Saved threshold: {self.saved_threshold_path}")

        print("\n=== RF PIPELINE COMPLETED ===")

    def run_inference(self):
        """Load pretrained RF + scaler + threshold and classify new dataset."""

        print("→ Loading pretrained model, scaler and threshold...")
        
        model = joblib.load(self.saved_model_path)

        threshold = float(np.load(self.saved_threshold_path))

        print("→ Loading dataset...")
        X_train, X_test, y_train, y_test, metadata_test, X_test_orig, X_err_test, _ = self.load_data(
            csv_path=self.csv_path, n_bins=self.n_bins, samples_per_class=self.samples_per_class, scaler=False
        )

        self.plot_confusion_matrix_image(y_test, y_pred_opt, best_thresh, save_path='confusion_matrix_rf.png')
        self.plot_roc_curve(y_test, proba_test, save_path='roc_curve_rf.png')
        self.plot_pr_curve(y_test, proba_test, save_path='pr_curve_rf.png')
        self.plot_probability_histograms(y_test, proba_test, save_path='probability_hist_RF.png')

        y_pred_opt, proba_test, best_thresh, roc_tuple = self.evaluate_with_optimal_threshold(model, X_test, y_test)

        print("\n=== INFERENCE RESULTS ===")
        print(classification_report(y_test, y_pred_opt))
        print(confusion_matrix(y_test, y_pred_opt))


class CNNClassifier:

    def __init__(self, params: dict):

        # Read CNN-related parameters from "params.yaml"
        cnn_params = params.get("cnn", {})

        self.csv_path = cnn_params.get("dataset_path", "tess_data.csv")
        self.n_bins = int(cnn_params.get("n_bins", 1000))
        self.samples_per_class = int(cnn_params.get("samples_per_class", 350))
        self.learning_rate = float(cnn_params.get("learning_rate", 0.0001))
        self.epochs = int(cnn_params.get("epochs", 200))
        self.results_dir = cnn_params.get("results_dir", "./cnn_results")

        if self.results_dir is None or not os.path.exists(self.results_dir):
            print("CNN results_dir not specified or does not exist, please input a valid path in params.yaml")
            exit()
        
        self.saved_model_path = os.path.join(self.results_dir, "cnn_model.keras")
        self.saved_scaler_path = os.path.join(self.results_dir, "scaler.pkl")
        self.saved_threshold_path = os.path.join(self.results_dir, "threshold.npy")

        os.makedirs(self.results_dir, exist_ok=True)

        # Reproducibility
        np.random.seed(27)
        tf.random.set_seed(27)

        print("=" * 70)
        print("FINAL TESS CLASSIFICATION (CLI MODE)")
        print("=" * 70)
        print("TF version:", tf.__version__)
    
    def artifacts_exist(self):
        """Check if pretrained model, scaler, and threshold already exist."""
        return (
            os.path.exists(self.saved_model_path) and
            os.path.exists(self.saved_scaler_path) and
            os.path.exists(self.saved_threshold_path)
        )

    #  FOCAL LOSS 
    def focal_loss(self, gamma=2.5, alpha=0.75):
        """Focal loss optimized for severe imbalance (binary)."""

        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            focal_weight = alpha_factor * K.pow(1 - pt, gamma)
            bce = -K.log(pt)
            return K.mean(focal_weight * bce)

        return focal_loss_fixed
    
    #  DATA BALANCING 
    def create_balanced_dataset(self, X, y, samples_per_class=400):
        """Create a perfectly balanced dataset via lightweight augmentations."""
        print("\n" + "="*70)
        print("CREATING BALANCED DATASET")
        print("="*70)

        X_class0 = X[y == 0]
        X_class1 = X[y == 1]

        print(f"Original - Class 0: {len(X_class0)}, Class 1: {len(X_class1)}")

        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]

            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                # number we still need (cap to avoid oversampling too big chunks)
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))

                aug_type = np.random.rand()
                if aug_type < 0.25:
                    # Additive Gaussian noise
                    X_aug = X_orig[idx] + np.random.normal(
                        0, 0.002, (len(idx), X_orig.shape[1])
                    )
                elif aug_type < 0.5:
                    # Multiplicative scaling
                    scale = 1.0 + np.random.uniform(-0.01, 0.01, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    # Time shift
                    shifts = np.random.randint(-5, 5, len(idx))
                    X_aug = np.array(
                        [np.roll(X_orig[i], s) for i, s in zip(idx, shifts)]
                    )
                else:
                    # Mild combo: small scale + small noise
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)

                X_result.append(X_aug)

            X_final = np.vstack(X_result)
            return X_final[:n_target]

        X0_bal = augment_to_target(X_class0, samples_per_class)
        X1_bal = augment_to_target(X_class1, samples_per_class)

        print(f"Balanced - Class 0: {len(X0_bal)}, Class 1: {len(X1_bal)}")

        X_balanced = np.vstack([X0_bal, X1_bal])
        y_balanced = np.concatenate([
            np.zeros(samples_per_class),
            np.ones(samples_per_class)
        ])

        # Shuffle
        idx = np.arange(len(X_balanced))
        np.random.shuffle(idx)

        return X_balanced[idx], y_balanced[idx]


    #  BUILD CNN 
    def build_simple_cnn(self):
        """Simpler CNN to prevent overfitting on small datasets."""
        print("\n" + "=" * 70)
        print("BUILDING SIMPLIFIED CNN")
        print("=" * 70)

        model = models.Sequential([
        layers.Input(shape=(self.n_bins, 1)),

        # Feature extraction
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(512, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),

        # Classification head
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
        layers.Dropout(0.1),

        layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.focal_loss(gamma=2.5, alpha=0.75),
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        model.summary()
        print("\nUsing Focal Loss (gamma=2.5, alpha=0.75)")
        return model

    #  TRAINING 
    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Train the model with AUC-centric callbacks."""
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                patience=20,
                restore_best_weights=True,
                mode="max",
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode="max",
                verbose=1,
            ),
            ModelCheckpoint(
                os.path.join(self.results_dir, "best_model_final.keras"),
                monitor="val_auc",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    #  EVALUATION 
    def evaluate_with_optimal_threshold(self, model, X_test, y_test):
        """Find an optimal threshold from ROC (Youden's J) and evaluate."""
        print("\n" + "=" * 70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("=" * 70)

        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        print(f"\nOptimal threshold: {optimal_threshold:.4f} (default=0.5)")
        print(
            f"  At this threshold: TPR={tpr[optimal_idx]:.4f}, "
            f"FPR={fpr[optimal_idx]:.4f}"
        )

        # Predictions with optimal vs default thresholds
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        y_pred_default = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        acc_optimal = accuracy_score(y_test, y_pred_optimal)
        acc_default = accuracy_score(y_test, y_pred_default)
        auc = roc_auc_score(y_test, y_pred_proba)

        print("\nResults:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(
            f"  Accuracy (default threshold=0.5): {acc_default:.4f} ({acc_default*100:.2f}%)"
        )
        print(
            f"  Accuracy (optimal threshold={optimal_threshold:.4f}): "
            f"{acc_optimal:.4f} ({acc_optimal*100:.2f}%)"
        )

        print("\nWith optimal threshold:")
        print(
            classification_report(
                y_test,
                y_pred_optimal,
                target_names=["Non-Planet", "Planet"],
                digits=4,
                zero_division=0,
            )
        )

        print("\nPrediction distribution (optimal threshold):")
        print(f"  Predicted 0: {(y_pred_optimal == 0).sum()}")
        print(f"  Predicted 1: {(y_pred_optimal == 1).sum()}")
        print("True distribution:")
        print(f"  True 0: {(y_test == 0).sum()}")
        print(f"  True 1: {(y_test == 1).sum()}")

        return y_pred_optimal, y_pred_proba, optimal_threshold
    
    #  PLOT INDIVIDUAL LIGHT CURVES WITH PREDICTIONS
    def plot_lightcurves_with_predictions(
        self,
        X_test_orig,
        X_err_test,
        y_test,
        y_pred,
        y_pred_proba,
        metadata_test,
        scaler,
        threshold,
        n_samples=6,
        save_name="sample_lightcurves_predictions.png",
    ):
        """Plot light curves with error bars + prediction labels."""

        os.makedirs(self.results_dir, exist_ok=True)
        save_path = os.path.join(self.results_dir, save_name)

        print("\n" + "="*70)
        print(f"PLOTTING LIGHTCURVES WITH PREDICTIONS (n={n_samples})")
        print("="*70)

        n_samples = min(n_samples, len(X_test_orig))

        # Select diverse samples
        correct_planet = np.where((y_test == 1) & (y_pred == 1))[0]
        incorrect_planet = np.where((y_test == 1) & (y_pred == 0))[0]
        correct_nonplanet = np.where((y_test == 0) & (y_pred == 0))[0]
        incorrect_nonplanet = np.where((y_test == 0) & (y_pred == 1))[0]

        selected_idx = []
        per_category = max(1, n_samples // 4)

        for group in [correct_planet, incorrect_planet, correct_nonplanet, incorrect_nonplanet]:
            if len(group) > 0:
                take = min(per_category, len(group))
                selected_idx.extend(np.random.choice(group, take, replace=False))

        while len(selected_idx) < n_samples:
            remaining = list(set(range(len(y_test))) - set(selected_idx))
            if not remaining:
                break
            selected_idx.append(np.random.choice(remaining))

        selected_idx = np.array(selected_idx[:n_samples])

        # Figure layout
        n_cols = 2
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for plot_i, idx in enumerate(selected_idx):
            ax = axes[plot_i]

            flux_norm = X_test_orig[idx].flatten()
            flux_err = X_err_test[idx]
            flux_original = scaler.inverse_transform(flux_norm.reshape(1, -1)).flatten()

            time_bins = np.arange(len(flux_original))

            # Metadata 
            toi = metadata_test.loc[idx, "toi_name"]
            tic = metadata_test.loc[idx, "tic"]
            disp = metadata_test.loc[idx, "disp"]
            sector = metadata_test.loc[idx, "sector"]

            true_lbl = y_test[idx]
            pred_lbl = y_pred[idx]
            p = y_pred_proba[idx]

            is_correct = (true_lbl == pred_lbl)
            true_str = "Transit" if true_lbl == 1 else "Non-Transit"
            pred_str = "Transit" if pred_lbl == 1 else "Non-Transit"
            color = "green" if is_correct else "red"
            symbol = "✓" if is_correct else "✗"

            # Plot curve
            ax.errorbar(
                time_bins, flux_original, yerr=flux_err, fmt="o", markersize=2,
                ecolor="gray", elinewidth=0.5, alpha=0.6, label="Data"
            )

            baseline = np.median(flux_original)
            ax.axhline(baseline, linestyle="--", linewidth=1, alpha=0.7)

            ax.set_xlabel("Time Bin")
            ax.set_ylabel("Flux")
            ax.grid(alpha=0.3)

            ax.set_title(
                f"TOI {toi} (TIC {tic}, {disp}) - Sector {sector}\n"
                f"True: {true_str} | Pred: {pred_str} (p={p:.3f}) {symbol}",
                fontsize=10,
                color=color
            )

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        for j in range(n_samples, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            f"Sample Light-curve Predictions (Threshold={threshold:.3f})",
            fontsize=14, fontweight="bold", y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    #  PLOTTING 
    def plot_all(
            self,
            y_test,
            y_pred,
            y_pred_proba,
            history,
            threshold,
            metadata_test,
            X_test_orig,
            X_err_test,
            scaler,
    ):        
        """Generate main plots: confusion matrix + training curves."""

        print("\n" + "=" * 70)
        print("VISUALIZATIONS")
        print("=" * 70)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=["Non-Planet", "Planet"],
            yticklabels=["Non-Planet", "Planet"],
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix (threshold={threshold:.3f})",
        )

        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(
                    j,
                    i,
                    f"{count}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        plt.tight_layout()
        cm_path = os.path.join(self.results_dir, "confusion_matrix_final.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Saved: {cm_path}")
        plt.close()

        # Training history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC"), ("recall", "Recall")]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric in history.history and f"val_{metric}" in history.history:
                ax.plot(history.history[metric], label="Train", linewidth=2)
                ax.plot(history.history[f"val_{metric}"], label="Val", linewidth=2)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(title)
                ax.set_title(f"{title} vs Epoch", fontweight="bold")
                ax.legend()
                ax.grid(alpha=0.3)

        plt.suptitle("Training History - Final Model", fontsize=14, fontweight="bold")
        plt.tight_layout()
        hist_path = os.path.join(self.results_dir, "training_history_final.png")
        plt.savefig(hist_path, dpi=300)
        print(f"Saved: {hist_path}")
        plt.close()

        # Light curves
        self.plot_lightcurves_with_predictions(
            X_test_orig,
            X_err_test,
            y_test,
            y_pred,
            y_pred_proba,
            metadata_test,
            scaler,
            threshold,
        )

    # MAIN ENTRY POINT 
    def run(self):
        """
        If pretrained artifacts exist:
             load them and classify
        Else:
             train CNN, save artifacts, classify
        """

        # TRAIN-ONCE 
        if self.artifacts_exist():
            print("\n=== Pretrained model found — running inference ===\n")
            return self.run_inference()

        print("\n=== No pretrained CNN found — training new model ===\n")
        
        """
        Full pipeline:
        - load data from CSV
        - build & train CNN
        - find optimal threshold
        - plot statistics
        - write report_cnn_assignment2_taskF.txt
        """

        print("\n" + "=" * 70)
        print("LOADING & PREPARING DATA")
        print("=" * 70)

        X_train, X_test, y_train, y_test, metadata_test, X_test_orig, X_err_test, scaler = load_data(
            csv_path=self.csv_path, n_bins=self.n_bins, samples_per_class=self.samples_per_class, use_scaler=True
        )

        # Build model
        model = self.build_simple_cnn()

        # Train
        history = self.train_model(model, X_train, y_train, X_test, y_test)

        # Evaluate with optimal threshold
        y_pred, y_pred_proba, threshold = self.evaluate_with_optimal_threshold(
            model, X_test, y_test
        )

        # Plot stats (confusion matrix + history + sample lightcurves)
        self.plot_all(
            y_test,
            y_pred,
            y_pred_proba,
            history,
            threshold,
            metadata_test,
            X_test_orig,
            X_err_test,
            scaler,
        )

        # Write text report (confusion matrix + precision)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        report_path = os.path.join(
            self.results_dir, "report_cnn_assignment2_taskF.txt"
        )
        with open(report_path, "w") as f:
            f.write("CNN DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(
                classification_report(
                    y_test,
                    y_pred,
                    target_names=["Non-Planet", "Planet"],
                    digits=4,
                    zero_division=0,
                )
            )
            f.write("\n\nConfusion matrix:\n")
            f.write(str(cm))
            f.write(
                f"\n\nPrecision (optimal threshold={threshold:.4f}): {precision:.4f}\n"
            )

        print(f"\nSaved text report to {report_path}")

        # SAVE MODEL, SCALER, THRESHOLD FOR FUTURE USE
        print("\nSaving model, scaler and threshold for future runs...")
        model.save(self.saved_model_path)
        joblib.dump(scaler, self.saved_scaler_path)
        np.save(self.saved_threshold_path, threshold)

        print(f"Saved model: {self.saved_model_path}")
        print(f"Saved scaler: {self.saved_scaler_path}")
        print(f"Saved threshold: {self.saved_threshold_path}")

        print("\n=== CNN PIPELINE COMPLETED ===")

    def run_inference(self):
        """Load pretrained CNN + scaler + threshold and classify new dataset."""

        print("→ Loading pretrained model, scaler and threshold...")
        
        model = load_model(
            self.saved_model_path,
            custom_objects={"focal_loss_fixed": self.focal_loss()}
        )

        scaler = joblib.load(self.saved_scaler_path)
        threshold = float(np.load(self.saved_threshold_path))

        print("→ Loading dataset...")
        X_train, X_test, y_train, y_test, metadata_test, X_test_orig, X_err_test, _ = load_data(
            csv_path=self.csv_path, n_bins=self.n_bins, samples_per_class=self.samples_per_class, use_scaler=False
        )

        # Apply saved scaler
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = X_test_scaled.reshape(-1, self.n_bins, 1)

        print("→ Predicting using pretrained CNN...")
        y_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_proba >= threshold).astype(int)

        print("\n=== INFERENCE RESULTS ===")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        