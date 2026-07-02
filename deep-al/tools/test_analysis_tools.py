"""
Test script for federated learning analysis tools.
Creates sample data and tests the analysis pipeline.

Usage:
    python test_analysis_tools.py
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np


def create_sample_experiment(exp_dir: Path, num_rounds: int = 5, num_clients: int = 3, with_baseline: bool = True, config: dict = None):
    """Create a sample experiment directory with realistic data."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Default config if not provided
    if config is None:
        config = {
            "dataset": "cifar100",
            "al_method": "random",
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "partition_mode": "iid",
            "federated_mode": "standard"
        }
    
    # Generate global metrics
    global_metrics = []
    for round_id in range(num_rounds):
        # Simulate improving accuracy over rounds
        base_acc = 60 + round_id * 5 + np.random.randn() * 2
        global_acc = base_acc + np.random.randn() * 1
        
        round_metrics = {
            "round": round_id,
            "num_selected_clients": num_clients,
            "avg_client_acc": float(np.clip(base_acc, 0, 100)),
            "avg_client_loss": float(2.0 - round_id * 0.3 + np.random.randn() * 0.1),
            "global_test_acc": float(np.clip(global_acc, 0, 100)),
            "avg_num_labeled": float(100 + round_id * 50),
            "avg_num_veracity_used": float(round_id * 30),
            "avg_num_veracity_filtered": float(round_id * 5),
            "train_class_distribution": {str(i): 500 for i in range(10)},
            "test_class_distribution": {str(i): 100 for i in range(10)},
            "client_metrics": {
                str(client_id): {
                    "test_acc": float(base_acc + np.random.randn() * 3),
                    "train_loss": float(2.0 - round_id * 0.3),
                    "num_labeled": int(100 + round_id * 50),
                    "num_veracity_used": int(round_id * 30),
                    "num_veracity_filtered": int(round_id * 5)
                }
                for client_id in range(num_clients)
            }
        }
        global_metrics.append(round_metrics)
        
        # Create round directory with metrics
        round_dir = exp_dir / f"round_{round_id}"
        round_dir.mkdir(exist_ok=True)
        with open(round_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(round_metrics, f, indent=2)
    
    # Save global metrics
    with open(exp_dir / "global_metrics.json", "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)
    
    # Create baseline metrics if requested
    if with_baseline:
        baseline_metrics = {
            "avg_baseline_acc": 55.0 + np.random.randn() * 2,
            "client_baseline_metrics": {
                str(client_id): {
                    "test_acc": 55.0 + np.random.randn() * 3,
                    "train_loss": 2.5
                }
                for client_id in range(num_clients)
            }
        }
        with open(exp_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2)
    
    # Create dataset class distributions
    distributions = {
        "train_class_distribution": {str(i): 500 + int(np.random.randn() * 50) for i in range(10)},
        "test_class_distribution": {str(i): 100 + int(np.random.randn() * 10) for i in range(10)},
        "per_client_partition_distributions": {
            str(client_id): {
                str(cls): int(500 / num_clients + np.random.randn() * 20)
                for cls in range(10)
            }
            for client_id in range(num_clients)
        }
    }
    with open(exp_dir / "dataset_class_distributions.json", "w", encoding="utf-8") as f:
        json.dump(distributions, f, indent=2)
    
    # Save config
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created sample experiment: {exp_dir.name}")


def test_analysis_pipeline():
    """Test the complete analysis pipeline."""
    print("=" * 60)
    print("Testing Federated Learning Analysis Tools")
    print("=" * 60)
    
    # Create temporary directory for test data
    temp_dir = Path(tempfile.mkdtemp(prefix="federated_test_"))
    print(f"\nCreated temporary directory: {temp_dir}")
    
    try:
        # Create sample experiments with diverse configs
        print("\n1. Creating sample experiments...")
        experiments = [
            ("experiment_iid_baseline", 5, 3, True, {
                "dataset": "cifar100", "al_method": "random", "num_clients": 3, "num_rounds": 5,
                "partition_mode": "iid", "federated_mode": "standard", "local_epochs": 200,
                "clients_per_round": 3, "fl_method": "fedavg"
            }),
            ("experiment_dirichlet_low", 6, 4, True, {
                "dataset": "cifar100", "al_method": "bayes_misp", "num_clients": 4, "num_rounds": 6,
                "partition_mode": "dirichlet", "dirichlet_alpha": "0.1", "federated_mode": "standard",
                "local_epochs": 200, "clients_per_round": 4, "fl_method": "fedavg"
            }),
            ("experiment_veracity_threshold_low", 7, 5, True, {
                "dataset": "cifar100", "al_method": "bayes_misp", "num_clients": 5, "num_rounds": 7,
                "partition_mode": "dirichlet", "dirichlet_alpha": "0.5", "federated_mode": "veracity_query",
                "queries_per_round": 100, "veracity_threshold": "0.25", "veracity_loss_weight": "2",
                "local_epochs": 200, "clients_per_round": 5, "fl_method": "fedavg"
            }),
            ("experiment_veracity_threshold_high", 7, 5, True, {
                "dataset": "cifar100", "al_method": "bayes_misp", "num_clients": 5, "num_rounds": 7,
                "partition_mode": "dirichlet", "dirichlet_alpha": "0.5", "federated_mode": "veracity_query",
                "queries_per_round": 100, "veracity_threshold": "0.75", "veracity_loss_weight": "4",
                "local_epochs": 200, "clients_per_round": 5, "fl_method": "fedavg"
            }),
            ("experiment_fedprox", 5, 10, True, {
                "dataset": "cifar100", "al_method": "random", "num_clients": 10, "num_rounds": 5,
                "partition_mode": "dirichlet", "dirichlet_alpha": "1.0", "federated_mode": "standard",
                "local_epochs": 200, "clients_per_round": 10, "fl_method": "fedprox", "fedprox_mu": "0.01"
            }),
        ]
        
        for exp_name, num_rounds, num_clients, with_baseline, config in experiments:
            exp_dir = temp_dir / exp_name
            create_sample_experiment(exp_dir, num_rounds, num_clients, with_baseline, config)
        
        print(f"\nCreated {len(experiments)} sample experiments")
        
        # Test the parser
        print("\n2. Testing FederatedResultsParser...")
        from analyze_federated_results import FederatedResultsParser, generate_summary_statistics
        
        parser = FederatedResultsParser(str(temp_dir))
        exp_dirs = parser.find_experiment_dirs()
        print(f"✓ Found {len(exp_dirs)} experiment directories")
        
        parsed_experiments = parser.parse_all_experiments()
        print(f"✓ Successfully parsed {len(parsed_experiments)} experiments")
        
        # Test summary generation
        print("\n3. Testing summary statistics...")
        summary = generate_summary_statistics(parsed_experiments)
        print(f"✓ Generated summary statistics:")
        print(f"  - Total experiments: {summary['total_experiments']}")
        print(f"  - Avg final client acc: {summary['avg_final_client_acc']:.2f}%")
        print(f"  - Avg final global acc: {summary['avg_final_global_acc']:.2f}%")
        if summary.get('avg_baseline_acc'):
            print(f"  - Avg baseline acc: {summary['avg_baseline_acc']:.2f}%")
            print(f"  - Avg improvement: {summary['avg_improvement_over_baseline']:.2f}%")
        
        # Save analysis results
        print("\n4. Saving analysis results...")
        output_path = temp_dir / "test_analysis.json"
        output_data = {
            "summary": summary,
            "experiments": [
                {
                    "exp_name": exp.exp_name,
                    "exp_path": exp.exp_path,
                    "num_rounds": exp.num_rounds,
                    "num_clients": exp.num_clients,
                    "final_avg_client_acc": exp.final_avg_client_acc,
                    "final_global_test_acc": exp.final_global_test_acc,
                    "best_avg_client_acc": exp.best_avg_client_acc,
                    "best_global_test_acc": exp.best_global_test_acc,
                    "baseline_acc": exp.baseline_acc,
                    "improvement_over_baseline": exp.improvement_over_baseline,
                    "avg_labeled_samples": exp.avg_labeled_samples,
                    "avg_veracity_used": exp.avg_veracity_used,
                    "avg_veracity_filtered": exp.avg_veracity_filtered,
                    "rounds_data": exp.rounds_data,
                    "train_class_distribution": exp.train_class_distribution,
                    "test_class_distribution": exp.test_class_distribution,
                }
                for exp in parsed_experiments
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved analysis to: {output_path}")
        
        # Verify JSON can be loaded
        print("\n5. Verifying JSON output...")
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print(f"✓ JSON is valid and loadable")
        print(f"  - Contains {len(loaded_data['experiments'])} experiments")
        print(f"  - Summary has {len(loaded_data['summary'])} fields")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
        print(f"\nTest data location: {temp_dir}")
        print(f"Analysis output: {output_path}")
        print("\nYou can now test the dashboard with:")
        print(f"  python view_federated_dashboard.py {output_path}")
        
        # Ask if user wants to keep the data
        print("\n" + "-" * 60)
        response = input("Keep test data? (y/n): ").lower().strip()
        
        if response != 'y':
            print("Cleaning up test data...")
            shutil.rmtree(temp_dir)
            print("✓ Test data removed")
        else:
            print(f"✓ Test data preserved at: {temp_dir}")
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if temp_dir.exists():
            print("\nCleaning up test data...")
            shutil.rmtree(temp_dir)
        
        return False
    
    return True


if __name__ == "__main__":
    import sys
    success = test_analysis_pipeline()
    sys.exit(0 if success else 1)
