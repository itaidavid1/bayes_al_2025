"""
Script to extract and analyze federated learning experiment results.
Parses output from FederatedServer runs and generates comprehensive metrics.

Usage:
    python analyze_federated_results.py <results_dir> [--output output.json]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""
    exp_name: str
    exp_path: str
    num_rounds: int
    num_clients: int
    
    # Overall metrics
    final_avg_client_acc: float
    final_global_test_acc: float
    best_avg_client_acc: float
    best_global_test_acc: float
    
    # Baseline comparison
    baseline_acc: Optional[float] = None
    improvement_over_baseline: Optional[float] = None
    
    # Active learning metrics
    avg_labeled_samples: List[float] = None
    avg_veracity_used: List[float] = None
    avg_veracity_filtered: List[float] = None
    
    # Per-round metrics
    rounds_data: List[Dict[str, Any]] = None
    
    # Dataset info
    train_class_distribution: Optional[Dict[str, int]] = None
    test_class_distribution: Optional[Dict[str, int]] = None
    per_client_distributions: Optional[Dict[str, Dict[str, int]]] = None
    
    # Config info
    config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.avg_labeled_samples is None:
            self.avg_labeled_samples = []
        if self.avg_veracity_used is None:
            self.avg_veracity_used = []
        if self.avg_veracity_filtered is None:
            self.avg_veracity_filtered = []
        if self.rounds_data is None:
            self.rounds_data = []


class FederatedResultsParser:
    """Parser for federated learning experiment results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")
    
    def find_experiment_dirs(self) -> List[Path]:
        """Find all experiment directories containing global_metrics.json."""
        exp_dirs = []
        
        for root, dirs, files in os.walk(self.results_dir):
            if "global_metrics.json" in files:
                exp_dirs.append(Path(root))
        
        return sorted(exp_dirs)
    
    def load_json_file(self, filepath: Path) -> Optional[Dict]:
        """Load JSON file safely."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None
    
    def parse_experiment(self, exp_dir: Path) -> Optional[ExperimentMetrics]:
        """Parse a single experiment directory."""
        # Load global metrics (required)
        global_metrics_path = exp_dir / "global_metrics.json"
        global_metrics = self.load_json_file(global_metrics_path)
        
        if not global_metrics:
            print(f"Skipping {exp_dir}: Could not load global_metrics.json")
            return None
        
        # Load baseline metrics (optional)
        baseline_metrics_path = exp_dir / "baseline_metrics.json"
        baseline_metrics = self.load_json_file(baseline_metrics_path)
        
        # Load dataset distributions (optional)
        dist_path = exp_dir / "dataset_class_distributions.json"
        distributions = self.load_json_file(dist_path)
        
        # Load config if available (optional)
        config_path = exp_dir / "config.json"
        config = self.load_json_file(config_path)
        
        # Extract metrics
        num_rounds = len(global_metrics)
        if num_rounds == 0:
            print(f"Skipping {exp_dir}: No rounds found")
            return None
        
        # Get final and best metrics
        avg_client_accs = [r.get("avg_client_acc", 0) for r in global_metrics]
        global_test_accs = [r.get("global_test_acc", 0) for r in global_metrics]
        
        final_avg_client_acc = avg_client_accs[-1] if avg_client_accs else 0
        final_global_test_acc = global_test_accs[-1] if global_test_accs else 0
        best_avg_client_acc = max(avg_client_accs) if avg_client_accs else 0
        best_global_test_acc = max(global_test_accs) if global_test_accs else 0
        
        # Extract baseline
        baseline_acc = None
        if baseline_metrics:
            baseline_acc = baseline_metrics.get("avg_baseline_acc")
        
        improvement = None
        if baseline_acc is not None:
            improvement = final_avg_client_acc - baseline_acc
        
        # Extract per-round data
        avg_labeled = [r.get("avg_num_labeled", 0) for r in global_metrics]
        avg_veracity = [r.get("avg_num_veracity_used", 0) for r in global_metrics]
        avg_veracity_filtered = [r.get("avg_num_veracity_filtered", 0) for r in global_metrics]
        
        # Get number of clients from first round
        num_clients = global_metrics[0].get("num_selected_clients", 0)
        
        # Extract distribution info
        train_dist = None
        test_dist = None
        per_client_dist = None
        if distributions:
            train_dist = distributions.get("train_class_distribution")
            test_dist = distributions.get("test_class_distribution")
            per_client_dist = distributions.get("per_client_partition_distributions")
        
        exp_name = exp_dir.name
        
        return ExperimentMetrics(
            exp_name=exp_name,
            exp_path=str(exp_dir),
            num_rounds=num_rounds,
            num_clients=num_clients,
            final_avg_client_acc=final_avg_client_acc,
            final_global_test_acc=final_global_test_acc,
            best_avg_client_acc=best_avg_client_acc,
            best_global_test_acc=best_global_test_acc,
            baseline_acc=baseline_acc,
            improvement_over_baseline=improvement,
            avg_labeled_samples=avg_labeled,
            avg_veracity_used=avg_veracity,
            avg_veracity_filtered=avg_veracity_filtered,
            rounds_data=global_metrics,
            train_class_distribution=train_dist,
            test_class_distribution=test_dist,
            per_client_distributions=per_client_dist,
            config=config
        )
    
    def parse_all_experiments(self) -> List[ExperimentMetrics]:
        """Parse all experiments in the results directory."""
        exp_dirs = self.find_experiment_dirs()
        print(f"Found {len(exp_dirs)} experiment directories")
        
        experiments = []
        for exp_dir in exp_dirs:
            print(f"\nParsing: {exp_dir}")
            exp_metrics = self.parse_experiment(exp_dir)
            if exp_metrics:
                experiments.append(exp_metrics)
                print(f"  ✓ Parsed successfully")
                print(f"    Rounds: {exp_metrics.num_rounds}")
                print(f"    Final Avg Acc: {exp_metrics.final_avg_client_acc:.2f}%")
                if exp_metrics.baseline_acc is not None:
                    print(f"    Baseline: {exp_metrics.baseline_acc:.2f}%")
                    print(f"    Improvement: {exp_metrics.improvement_over_baseline:.2f}%")
        
        return experiments


def extract_filter_values(experiments: List[ExperimentMetrics]) -> Dict[str, List]:
    """Extract unique values for each filterable parameter across all experiments."""
    filter_values = {}
    
    # Parameters to extract from config
    config_params = [
        'dataset', 'al_method', 'partition_mode', 'dirichlet_alpha',
        'federated_mode', 'queries_per_round', 'veracity_threshold',
        'veracity_loss_weight', 'clients_per_round', 'num_rounds',
        'local_epochs', 'fl_method', 'fedprox_mu', 'client_labels_initial_size',
        'eval_model', 'diff_method', 'cont_method', 'kernel_type'
    ]
    
    for param in config_params:
        values = set()
        for exp in experiments:
            if exp.config and param in exp.config:
                val = exp.config[param]
                # Convert to string for consistency
                if val is not None:
                    values.add(str(val))
        
        if values:
            # Sort values (numerically if possible, otherwise alphabetically)
            try:
                sorted_values = sorted(values, key=lambda x: float(x) if x.replace('.', '').replace('-', '').isdigit() else x)
            except:
                sorted_values = sorted(values)
            filter_values[param] = sorted_values
    
    return filter_values


def generate_summary_statistics(experiments: List[ExperimentMetrics]) -> Dict[str, Any]:
    """Generate summary statistics across all experiments."""
    if not experiments:
        return {}
    
    summary = {
        "total_experiments": len(experiments),
        "timestamp": datetime.now().isoformat(),
        "experiments_with_baseline": sum(1 for e in experiments if e.baseline_acc is not None),
        "avg_final_client_acc": float(np.mean([e.final_avg_client_acc for e in experiments])),
        "avg_final_global_acc": float(np.mean([e.final_global_test_acc for e in experiments])),
        "best_experiment": None,
        "worst_experiment": None,
    }
    
    # Find best and worst experiments
    best_exp = max(experiments, key=lambda e: e.final_avg_client_acc)
    worst_exp = min(experiments, key=lambda e: e.final_avg_client_acc)
    
    summary["best_experiment"] = {
        "name": best_exp.exp_name,
        "final_acc": best_exp.final_avg_client_acc,
        "path": best_exp.exp_path
    }
    
    summary["worst_experiment"] = {
        "name": worst_exp.exp_name,
        "final_acc": worst_exp.final_avg_client_acc,
        "path": worst_exp.exp_path
    }
    
    # Baseline comparisons
    exps_with_baseline = [e for e in experiments if e.baseline_acc is not None]
    if exps_with_baseline:
        summary["avg_baseline_acc"] = float(np.mean([e.baseline_acc for e in exps_with_baseline]))
        summary["avg_improvement_over_baseline"] = float(np.mean([e.improvement_over_baseline for e in exps_with_baseline]))
    
    # Add filter values
    summary["filter_values"] = extract_filter_values(experiments)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze federated learning experiment results"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="/cs/labs/daphna/itai.david/py_repos/TypiClust/federated_results/federated_analysis.json",
        help="Output JSON file path (default: federated_analysis.json)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse experiments
    print("="*60)
    print("Federated Learning Results Parser")
    print("="*60)


    results_dir = "/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100_dino/resnet18/federated/2026_05_11/"
    # filter folders

    parser_obj = FederatedResultsParser(results_dir)
    experiments = parser_obj.parse_all_experiments()
    
    if not experiments:
        print("\nNo valid experiments found!")
        return
    
    # Generate summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    summary = generate_summary_statistics(experiments)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Average final client accuracy: {summary['avg_final_client_acc']:.2f}%")
    print(f"Average final global accuracy: {summary['avg_final_global_acc']:.2f}%")
    
    if summary.get('avg_baseline_acc'):
        print(f"\nBaseline comparison:")
        print(f"  Average baseline accuracy: {summary['avg_baseline_acc']:.2f}%")
        print(f"  Average improvement: {summary['avg_improvement_over_baseline']:.2f}%")
    
    print(f"\nBest experiment: {summary['best_experiment']['name']}")
    print(f"  Accuracy: {summary['best_experiment']['final_acc']:.2f}%")
    
    # Save results
    output_data = {
        "summary": summary,
        "experiments": [asdict(exp) for exp in experiments]
    }
    
    output_path = Path(args.output)
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path.absolute()}")
    print(f"\nTo view the interactive dashboard, run:")
    print(f"  python view_federated_dashboard.py {output_path}")


if __name__ == "__main__":
    main()
