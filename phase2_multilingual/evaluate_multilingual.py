"""
Phase 2: Multilingual Evaluation Benchmarks

Comprehensive evaluation across languages, code-mix types, and scenarios.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualEvaluator:
    """
    Comprehensive evaluation for multilingual content moderation.
    
    Metrics:
    - Per-language performance
    - Code-mix detection accuracy
    - Cross-lingual transfer effectiveness
    - Routing accuracy
    """
    
    def __init__(self, output_dir: str = "evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_per_language(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        languages: List[str],
        language_names: List[str] = None
    ) -> Dict:
        """
        Evaluate model performance per language.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            languages: Language for each sample
            language_names: Optional list of language names to evaluate
        
        Returns:
            Dictionary with per-language metrics
        """
        if language_names is None:
            language_names = list(set(languages))
        
        results = {}
        
        for lang in language_names:
            lang_mask = np.array([l == lang for l in languages])
            
            if lang_mask.sum() == 0:
                logger.warning(f"No samples for language: {lang}")
                continue
            
            lang_true = y_true[lang_mask]
            lang_pred = y_pred[lang_mask]
            lang_prob = y_prob[lang_mask]
            
            # Skip if no positive samples
            if lang_true.sum() == 0:
                logger.warning(f"No positive samples for {lang}")
                continue
            
            # Classification report
            report = classification_report(
                lang_true, lang_pred,
                output_dict=True,
                zero_division=0
            )
            
            # AUC
            try:
                lang_auc = auc(*roc_curve(lang_true, lang_prob)[:2])
            except:
                lang_auc = 0.0
            
            results[lang] = {
                'count': int(lang_mask.sum()),
                'positive_count': int(lang_true.sum()),
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score'],
                'auc': float(lang_auc)
            }
        
        return results
    
    def evaluate_code_mix_detection(
        self,
        true_code_mixed: List[bool],
        detected_code_mixed: List[bool],
        code_mix_types_true: List[str],
        code_mix_types_detected: List[str]
    ) -> Dict:
        """
        Evaluate code-mix detection accuracy.
        
        Returns:
            Dictionary with detection metrics
        """
        # Binary: is code-mixed or not
        true_binary = np.array(true_code_mixed).astype(int)
        detected_binary = np.array(detected_code_mixed).astype(int)
        
        report = classification_report(
            true_binary, detected_binary,
            target_names=['not_code_mixed', 'code_mixed'],
            output_dict=True,
            zero_division=0
        )
        
        # Type-specific accuracy (only for code-mixed samples)
        code_mixed_mask = true_binary == 1
        if code_mixed_mask.sum() > 0:
            type_true = np.array(code_mix_types_true)[code_mixed_mask]
            type_detected = np.array(code_mix_types_detected)[code_mixed_mask]
            
            # Remove None values
            valid_mask = (type_true != None) & (type_detected != None)
            if valid_mask.sum() > 0:
                type_accuracy = (type_true[valid_mask] == type_detected[valid_mask]).mean()
            else:
                type_accuracy = 0.0
        else:
            type_accuracy = 0.0
        
        return {
            'detection_accuracy': report['accuracy'],
            'detection_precision': report['code_mixed']['precision'],
            'detection_recall': report['code_mixed']['recall'],
            'detection_f1': report['code_mixed']['f1-score'],
            'type_accuracy': float(type_accuracy)
        }
    
    def evaluate_routing_accuracy(
        self,
        true_routes: List[str],
        predicted_routes: List[str]
    ) -> Dict:
        """
        Evaluate routing decision accuracy.
        
        Args:
            true_routes: Ground truth routing decisions ('phase1' or 'phase2')
            predicted_routes: Predicted routing decisions
        
        Returns:
            Routing accuracy metrics
        """
        true_routes_array = np.array(true_routes)
        pred_routes_array = np.array(predicted_routes)
        
        report = classification_report(
            true_routes_array,
            pred_routes_array,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(
            true_routes_array,
            pred_routes_array,
            labels=['phase1', 'phase2']
        )
        
        return {
            'accuracy': report['accuracy'],
            'phase1_precision': report.get('phase1', {}).get('precision', 0),
            'phase1_recall': report.get('phase1', {}).get('recall', 0),
            'phase2_precision': report.get('phase2', {}).get('precision', 0),
            'phase2_recall': report.get('phase2', {}).get('recall', 0),
            'confusion_matrix': cm.tolist()
        }
    
    def plot_per_language_performance(
        self,
        metrics: Dict,
        save_path: Path = None
    ):
        """Plot per-language performance comparison."""
        if not metrics:
            logger.warning("No metrics to plot")
            return
        
        # Prepare data
        languages = list(metrics.keys())
        f1_scores = [metrics[lang]['f1'] for lang in languages]
        auc_scores = [metrics[lang]['auc'] for lang in languages]
        counts = [metrics[lang]['count'] for lang in languages]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # F1 scores
        axes[0].barh(languages, f1_scores, color='skyblue')
        axes[0].set_xlabel('F1 Score')
        axes[0].set_title('F1 Score by Language')
        axes[0].set_xlim(0, 1)
        
        # Add count annotations
        for i, (lang, f1, count) in enumerate(zip(languages, f1_scores, counts)):
            axes[0].text(f1 + 0.02, i, f'n={count}', va='center', fontsize=9)
        
        # AUC scores
        axes[1].barh(languages, auc_scores, color='lightcoral')
        axes[1].set_xlabel('AUC-ROC')
        axes[1].set_title('AUC-ROC by Language')
        axes[1].set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.savefig(self.output_dir / 'per_language_performance.png', dpi=150)
        
        plt.close()
    
    def plot_language_comparison(
        self,
        metrics_dict: Dict[str, Dict],
        save_path: Path = None
    ):
        """
        Compare metrics across different model versions or configurations.
        
        Args:
            metrics_dict: Dictionary of {model_name: per_language_metrics}
        """
        # Extract data
        all_languages = set()
        for metrics in metrics_dict.values():
            all_languages.update(metrics.keys())
        
        languages = sorted(all_languages)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['f1', 'precision', 'recall', 'auc']
        titles = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx]
            
            x = np.arange(len(languages))
            width = 0.8 / len(metrics_dict)
            
            for i, (model_name, metrics) in enumerate(metrics_dict.items()):
                values = [metrics.get(lang, {}).get(metric, 0) for lang in languages]
                ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)
            
            ax.set_xlabel('Language')
            ax.set_ylabel(title)
            ax.set_title(f'{title} by Language')
            ax.set_xticks(x + width * (len(metrics_dict) - 1) / 2)
            ax.set_xticklabels(languages, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'language_comparison.png', dpi=150)
        
        plt.close()
    
    def generate_report(
        self,
        per_language_metrics: Dict,
        code_mix_metrics: Dict = None,
        routing_metrics: Dict = None,
        save_path: Path = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Markdown-formatted report
        """
        report = []
        report.append("# Multilingual Content Moderation - Evaluation Report\n")
        report.append("=" * 80 + "\n")
        
        # Per-language performance
        report.append("\n## Per-Language Performance\n")
        report.append("| Language | Count | F1 | Precision | Recall | AUC |\n")
        report.append("|----------|-------|-----|-----------|--------|-----|\n")
        
        for lang, metrics in sorted(per_language_metrics.items()):
            report.append(
                f"| {lang:15s} | {metrics['count']:5d} | "
                f"{metrics['f1']:.3f} | {metrics['precision']:.3f} | "
                f"{metrics['recall']:.3f} | {metrics['auc']:.3f} |\n"
            )
        
        # Overall statistics
        if per_language_metrics:
            all_f1 = [m['f1'] for m in per_language_metrics.values()]
            all_auc = [m['auc'] for m in per_language_metrics.values()]
            
            report.append(f"\n**Average F1**: {np.mean(all_f1):.3f} (±{np.std(all_f1):.3f})\n")
            report.append(f"**Average AUC**: {np.mean(all_auc):.3f} (±{np.std(all_auc):.3f})\n")
        
        # Code-mix evaluation
        if code_mix_metrics:
            report.append("\n## Code-Mix Detection Performance\n")
            report.append(f"- Detection Accuracy: {code_mix_metrics['detection_accuracy']:.3f}\n")
            report.append(f"- Detection F1: {code_mix_metrics['detection_f1']:.3f}\n")
            report.append(f"- Type Accuracy: {code_mix_metrics['type_accuracy']:.3f}\n")
        
        # Routing evaluation
        if routing_metrics:
            report.append("\n## Routing Accuracy\n")
            report.append(f"- Overall Accuracy: {routing_metrics['accuracy']:.3f}\n")
            report.append(f"- Phase 1 Precision: {routing_metrics['phase1_precision']:.3f}\n")
            report.append(f"- Phase 2 Precision: {routing_metrics['phase2_precision']:.3f}\n")
        
        report_text = ''.join(report)
        
        # Save report
        if save_path is None:
            save_path = self.output_dir / 'evaluation_report.md'
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Saved evaluation report to {save_path}")
        
        return report_text


def main():
    """Demo evaluation on sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multilingual model")
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions file')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels file')
    parser.add_argument('--languages', type=str, required=True,
                       help='Path to languages file')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    y_prob = np.load(args.predictions)
    y_true = np.load(args.labels)
    
    with open(args.languages, 'r') as f:
        languages = json.load(f)
    
    y_pred = (y_prob > 0.5).astype(int)
    
    # Evaluate
    evaluator = MultilingualEvaluator(args.output_dir)
    
    # Per-language metrics
    per_lang_metrics = evaluator.evaluate_per_language(
        y_true, y_pred, y_prob, languages
    )
    
    # Plot results
    evaluator.plot_per_language_performance(per_lang_metrics)
    
    # Generate report
    report = evaluator.generate_report(per_lang_metrics)
    
    print("\n" + report)


if __name__ == "__main__":
    main()
