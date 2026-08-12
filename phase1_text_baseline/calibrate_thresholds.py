"""
Phase 1.2: Calibrate Confidence Limits and Set Thresholds

This script takes the validation predictions from Step 1.1 and:
1. Calibrates the model outputs using temperature scaling or Platt scaling
2. Finds optimal classification thresholds for each label
3. Defines enforcement tier boundaries (auto-remove, human-review, auto-approve)
"""

import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score
)
from scipy.optimize import minimize
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature Scaling for model calibration.
    
    Scales the logits by a learned temperature parameter T:
    q_i = max_k exp(z_k / T) / sum_j exp(z_j / T)
    
    For binary classification, we optimize T to minimize NLL on validation set.
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Find optimal temperature using NLL loss.
        
        Args:
            logits: Raw model outputs before sigmoid (if available) or probabilities
            labels: Ground truth labels
        """
        # Convert probabilities back to logits if needed
        # (assuming input is probabilities from sigmoid)
        logits_torch = torch.FloatTensor(np.log(logits / (1 - logits + 1e-8)))
        labels_torch = torch.FloatTensor(labels)
        
        def nll_loss(T):
            """Negative log-likelihood loss."""
            scaled_logits = logits_torch / T
            probs = torch.sigmoid(scaled_logits)
            
            # Binary cross-entropy
            loss = -torch.mean(
                labels_torch * torch.log(probs + 1e-8) +
                (1 - labels_torch) * torch.log(1 - probs + 1e-8)
            )
            return loss.item()
        
        # Optimize temperature
        result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        
        logger.info(f"Optimal temperature: {self.temperature:.4f}")
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        logits_torch = torch.FloatTensor(np.log(logits / (1 - logits + 1e-8)))
        scaled_logits = logits_torch / self.temperature
        calibrated_probs = torch.sigmoid(scaled_logits).numpy()
        return calibrated_probs


class PlattScaling:
    """
    Platt Scaling for calibration.
    
    Fits a logistic regression on top of model outputs:
    P(y=1|f) = 1 / (1 + exp(A*f + B))
    """
    
    def __init__(self):
        self.A = 1.0
        self.B = 0.0
    
    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling parameters."""
        from sklearn.linear_model import LogisticRegression
        
        # Reshape for sklearn
        probs_reshape = probs.reshape(-1, 1)
        labels_reshape = labels.ravel()
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(probs_reshape, labels_reshape)
        
        self.A = lr.coef_[0, 0]
        self.B = lr.intercept_[0]
        
        logger.info(f"Platt scaling parameters: A={self.A:.4f}, B={self.B:.4f}")
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        return 1.0 / (1.0 + np.exp(self.A * probs + self.B))


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> Dict:
    """
    Compute calibration metrics.
    
    Returns:
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Reliability diagram data
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_size = mask.sum()
            
            error = abs(bin_acc - bin_conf)
            ece += (bin_size / len(y_true)) * error
            mce = max(mce, error)
    
    return {
        'ece': ece,
        'mce': mce,
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    save_path: Path,
    label_name: str = "toxic",
    n_bins: int = 15
):
    """Plot reliability diagram before and after calibration."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before calibration
    metrics_before = compute_calibration_metrics(y_true, y_prob_before, n_bins)
    axes[0].plot(metrics_before['prob_pred'], metrics_before['prob_true'], 
                 marker='o', label='Model')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title(f'Before Calibration (ECE: {metrics_before["ece"]:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # After calibration
    metrics_after = compute_calibration_metrics(y_true, y_prob_after, n_bins)
    axes[1].plot(metrics_after['prob_pred'], metrics_after['prob_true'], 
                 marker='o', label='Calibrated', color='green')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    axes[1].set_xlabel('Mean Predicted Probability')
    axes[1].set_ylabel('Fraction of Positives')
    axes[1].set_title(f'After Calibration (ECE: {metrics_after["ece"]:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Reliability Diagram - {label_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved reliability diagram to {save_path}")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
    
    Returns:
        Optimal threshold and metrics at that threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 scores
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    
    if metric == 'f1':
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        # Find threshold with highest precision while maintaining recall > 0.5
        mask = recalls[:-1] > 0.5
        if mask.any():
            optimal_idx = np.argmax(precisions[:-1] * mask)
        else:
            optimal_idx = np.argmax(precisions[:-1])
    elif metric == 'recall':
        # Find threshold with highest recall while maintaining precision > 0.5
        mask = precisions[:-1] > 0.5
        if mask.any():
            optimal_idx = np.argmax(recalls[:-1] * mask)
        else:
            optimal_idx = np.argmax(recalls[:-1])
    else:
        optimal_idx = np.argmax(f1_scores)
    
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return optimal_threshold, metrics


def find_enforcement_tiers(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision_high: float = 0.95,
    target_precision_low: float = 0.98
) -> Dict:
    """
    Find threshold boundaries for enforcement tiers.
    
    Tiers:
    - Auto-remove: probability > T_high (very confident it's toxic)
    - Human-review: T_low < probability < T_high (uncertain)
    - Auto-approve: probability < T_low (very confident it's safe)
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        target_precision_high: Target precision for auto-remove tier
        target_precision_low: Target precision for auto-approve tier (on negative class)
    
    Returns:
        Dictionary with tier thresholds and statistics
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find T_high: threshold for auto-remove (high precision on positives)
    high_precision_mask = precisions[:-1] >= target_precision_high
    if high_precision_mask.any():
        t_high_idx = np.where(high_precision_mask)[0][-1]  # Highest recall with target precision
        t_high = thresholds[t_high_idx]
    else:
        t_high = 0.95  # Default fallback
    
    # Find T_low: threshold for auto-approve (high precision on negatives)
    # This means low false negative rate
    # We want P(actually safe | predicted safe) >= target_precision_low
    # Which is equivalent to finding threshold where negative predictive value is high
    
    # Calculate negative predictive value for each threshold
    npvs = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        npv = tn / (tn + fn + 1e-8)
        npvs.append(npv)
    
    npvs = np.array(npvs)
    low_threshold_mask = npvs >= target_precision_low
    
    if low_threshold_mask.any():
        t_low_idx = np.where(low_threshold_mask)[0][0]  # Lowest threshold with target NPV
        t_low = thresholds[t_low_idx]
    else:
        t_low = 0.30  # Default fallback
    
    # Ensure T_low < T_high
    if t_low >= t_high:
        t_low = t_high - 0.15
    
    # Calculate statistics for each tier
    auto_remove_mask = y_prob >= t_high
    human_review_mask = (y_prob >= t_low) & (y_prob < t_high)
    auto_approve_mask = y_prob < t_low
    
    tiers = {
        't_high': float(t_high),
        't_low': float(t_low),
        'auto_remove': {
            'count': int(auto_remove_mask.sum()),
            'percentage': float(auto_remove_mask.mean() * 100),
            'precision': float(y_true[auto_remove_mask].mean()) if auto_remove_mask.any() else 0.0,
            'recall': float(y_true[auto_remove_mask].sum() / (y_true.sum() + 1e-8))
        },
        'human_review': {
            'count': int(human_review_mask.sum()),
            'percentage': float(human_review_mask.mean() * 100),
            'positive_rate': float(y_true[human_review_mask].mean()) if human_review_mask.any() else 0.0
        },
        'auto_approve': {
            'count': int(auto_approve_mask.sum()),
            'percentage': float(auto_approve_mask.mean() * 100),
            'false_negative_rate': float(y_true[auto_approve_mask].mean()) if auto_approve_mask.any() else 0.0
        }
    }
    
    return tiers


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: float,
    tier_thresholds: Dict,
    save_path: Path,
    label_name: str = "toxic"
):
    """Plot precision-recall curve with threshold markers."""
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision-Recall curve
    axes[0].plot(recalls[:-1], precisions[:-1], label='PR Curve')
    
    # Mark thresholds
    for thresh_val, color, label in [
        (tier_thresholds['t_high'], 'red', f'Auto-remove (T={tier_thresholds["t_high"]:.3f})'),
        (optimal_threshold, 'green', f'Optimal (T={optimal_threshold:.3f})'),
        (tier_thresholds['t_low'], 'orange', f'Auto-approve (T={tier_thresholds["t_low"]:.3f})')
    ]:
        idx = np.argmin(np.abs(thresholds - thresh_val))
        axes[0].scatter(recalls[idx], precisions[idx], c=color, s=100, zorder=5, label=label)
    
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall Curve - {label_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 vs Threshold
    axes[1].plot(thresholds, f1_scores, label='F1 Score')
    axes[1].axvline(tier_thresholds['t_high'], color='red', linestyle='--', 
                    label=f'T_high={tier_thresholds["t_high"]:.3f}')
    axes[1].axvline(optimal_threshold, color='green', linestyle='--',
                    label=f'T_opt={optimal_threshold:.3f}')
    axes[1].axvline(tier_thresholds['t_low'], color='orange', linestyle='--',
                    label=f'T_low={tier_thresholds["t_low"]:.3f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title(f'F1 Score vs Threshold - {label_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved threshold analysis to {save_path}")


def main(config_path: str):
    """Main calibration function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    
    # Load predictions and labels from Step 1.1
    model_dir = Path(config['training']['save_dir'])
    predictions = np.load(model_dir / 'val_predictions.npy')
    labels = np.load(model_dir / 'val_labels.npy')
    
    logger.info(f"Loaded predictions: {predictions.shape}")
    logger.info(f"Loaded labels: {labels.shape}")
    
    # Create output directory
    output_dir = model_dir / 'calibration'
    output_dir.mkdir(exist_ok=True)
    
    # Results storage
    calibration_results = {
        'method': config['calibration']['method'],
        'labels': {}
    }
    
    label_names = config['labels']['names']
    
    # Process each label
    for i, label_name in enumerate(label_names):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing label: {label_name}")
        logger.info(f"{'='*50}")
        
        y_true = labels[:, i]
        y_prob = predictions[:, i]
        
        # Skip if no positive samples
        if y_true.sum() == 0:
            logger.warning(f"No positive samples for {label_name}, skipping...")
            continue
        
        # Compute calibration metrics before
        metrics_before = compute_calibration_metrics(y_true, y_prob, 
                                                     config['calibration']['n_bins'])
        logger.info(f"Before calibration - ECE: {metrics_before['ece']:.4f}, "
                   f"MCE: {metrics_before['mce']:.4f}")
        
        # Apply calibration
        if config['calibration']['method'] == 'temperature_scaling':
            calibrator = TemperatureScaling()
        elif config['calibration']['method'] == 'platt_scaling':
            calibrator = PlattScaling()
        else:
            logger.error(f"Unknown calibration method: {config['calibration']['method']}")
            continue
        
        calibrator.fit(y_prob, y_true)
        y_prob_calibrated = calibrator.transform(y_prob)
        
        # Compute calibration metrics after
        metrics_after = compute_calibration_metrics(y_prob_calibrated, y_true,
                                                    config['calibration']['n_bins'])
        logger.info(f"After calibration  - ECE: {metrics_after['ece']:.4f}, "
                   f"MCE: {metrics_after['mce']:.4f}")
        
        # Plot reliability diagram
        plot_reliability_diagram(
            y_true, y_prob, y_prob_calibrated,
            output_dir / f'{label_name}_reliability.png',
            label_name,
            config['calibration']['n_bins']
        )
        
        # Find optimal threshold
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            y_true, y_prob_calibrated,
            config['thresholds']['optimization_metric']
        )
        
        logger.info(f"\nOptimal threshold: {optimal_threshold:.4f}")
        logger.info(f"  Precision: {threshold_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {threshold_metrics['recall']:.4f}")
        logger.info(f"  F1:        {threshold_metrics['f1']:.4f}")
        
        # Find enforcement tier thresholds
        tier_thresholds = find_enforcement_tiers(
            y_true, y_prob_calibrated,
            config['thresholds']['tiers']['auto_remove']['target_precision'],
            config['thresholds']['tiers']['auto_approve']['target_precision']
        )
        
        logger.info(f"\nEnforcement Tier Thresholds:")
        logger.info(f"  Auto-remove (T_high):  {tier_thresholds['t_high']:.4f}")
        logger.info(f"    - Volume: {tier_thresholds['auto_remove']['percentage']:.2f}%")
        logger.info(f"    - Precision: {tier_thresholds['auto_remove']['precision']:.4f}")
        logger.info(f"    - Recall: {tier_thresholds['auto_remove']['recall']:.4f}")
        
        logger.info(f"  Human-review (T_low to T_high):")
        logger.info(f"    - Volume: {tier_thresholds['human_review']['percentage']:.2f}%")
        logger.info(f"    - Positive rate: {tier_thresholds['human_review']['positive_rate']:.4f}")
        
        logger.info(f"  Auto-approve (T_low):   {tier_thresholds['t_low']:.4f}")
        logger.info(f"    - Volume: {tier_thresholds['auto_approve']['percentage']:.2f}%")
        logger.info(f"    - False negative rate: {tier_thresholds['auto_approve']['false_negative_rate']:.4f}")
        
        # Plot threshold analysis
        plot_threshold_analysis(
            y_true, y_prob_calibrated, optimal_threshold, tier_thresholds,
            output_dir / f'{label_name}_thresholds.png',
            label_name
        )
        
        # Store results
        calibration_results['labels'][label_name] = {
            'ece_before': float(metrics_before['ece']),
            'ece_after': float(metrics_after['ece']),
            'optimal_threshold': float(optimal_threshold),
            'threshold_metrics': threshold_metrics,
            'tier_thresholds': tier_thresholds,
            'calibration_params': {
                'temperature': calibrator.temperature if hasattr(calibrator, 'temperature') else None,
                'A': calibrator.A if hasattr(calibrator, 'A') else None,
                'B': calibrator.B if hasattr(calibrator, 'B') else None
            }
        }
    
    # Save calibration results
    import json
    with open(output_dir / 'calibration_results.json', 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("Calibration completed!")
    logger.info(f"Results saved to {output_dir / 'calibration_results.json'}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate model and find thresholds")
    parser.add_argument(
        "--config",
        type=str,
        default="phase1_text_baseline/configs/baseline.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)
