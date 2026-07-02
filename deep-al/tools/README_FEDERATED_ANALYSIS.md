# Federated Learning Results Analysis Tools

This directory contains tools for analyzing and visualizing federated learning experiment results from the FederatedServer outputs.

## Overview

The analysis toolkit consists of four main components:

1. **`analyze_federated_results.py`** - Extracts and aggregates metrics from experiment directories
2. **`generate_html_dashboard.py`** - Generates standalone HTML dashboard with interactive charts
3. **`view_federated_dashboard.py`** - Helper script to generate and launch the dashboard
4. **`test_analysis_tools.py`** - Testing script with sample data generation

## Quick Start

### Step 1: Extract Experiment Results

Run the analysis script on your results directory:

```bash
python analyze_federated_results.py /path/to/results/directory -o analysis_results.json
```

**Example:**
```bash
python analyze_federated_results.py ../experiments/federated_runs -o my_analysis.json
```

This will:
- Scan the directory for all experiment folders containing `global_metrics.json`
- Extract metrics from each experiment
- Generate summary statistics
- Save everything to a JSON file

### Step 2: View the Dashboard

Generate and open the interactive HTML dashboard:

```bash
python view_federated_dashboard.py analysis_results.json
```

Or generate the HTML manually:

```bash
python generate_html_dashboard.py analysis_results.json -o dashboard.html
```

Then open `dashboard.html` in any web browser.

## Detailed Usage

### Analysis Script Options

```bash
python analyze_federated_results.py [OPTIONS] RESULTS_DIR
```

**Arguments:**
- `RESULTS_DIR` - Path to directory containing federated experiment outputs

**Options:**
- `-o, --output FILE` - Output JSON file path (default: `federated_analysis.json`)
- `-v, --verbose` - Print verbose output during parsing

**Example with all options:**
```bash
python analyze_federated_results.py ~/experiments/federated_runs \
    --output detailed_analysis.json \
    --verbose
```

### Expected Directory Structure

The script expects experiment directories with the following structure:

```
results_directory/
├── experiment_1/
│   ├── global_metrics.json          # Required: Per-round metrics
│   ├── baseline_metrics.json        # Optional: Baseline comparison
│   ├── dataset_class_distributions.json  # Optional: Class distributions
│   ├── config.json                  # Optional: Experiment config
│   └── round_X/
│       └── metrics.json             # Individual round metrics
├── experiment_2/
│   ├── global_metrics.json
│   └── ...
└── ...
```

### Metrics Extracted

The analysis script extracts the following metrics for each experiment:

#### Overall Metrics
- **Final Average Client Accuracy** - Accuracy averaged across clients in final round
- **Final Global Test Accuracy** - Global model accuracy on test set in final round
- **Best Average Client Accuracy** - Highest client accuracy achieved across all rounds
- **Best Global Test Accuracy** - Highest global accuracy achieved across all rounds

#### Baseline Comparison (if available)
- **Baseline Accuracy** - Performance without veracity feedback
- **Improvement Over Baseline** - Percentage point improvement

#### Active Learning Metrics (per round)
- **Average Labeled Samples** - Mean number of labeled samples per client
- **Average Veracity Used** - Mean number of veracity feedback points consumed
- **Average Veracity Filtered** - Mean number of veracity points filtered out

#### Per-Round Data
- Client accuracy progression
- Global accuracy progression
- Training loss
- Number of selected clients
- Per-client detailed metrics

#### Dataset Information
- Train/test class distributions
- Per-client data partition distributions

## Dashboard Features

The interactive HTML dashboard provides:

### Filter Panel (All Tabs)
- **Dynamic Filters** - Automatically detects available config parameters
- **Multi-Select Filtering** - Filter by any combination of parameters:
  - Dataset (cifar10, cifar100, mnist)
  - AL Method (random, bayes_misp, etc.)
  - Partition Mode (iid, dirichlet)
  - Dirichlet Alpha (0.1, 0.5, 1.0, etc.)
  - Federated Mode (standard, veracity_query)
  - Veracity Parameters (threshold, loss_weight, queries_per_round)
  - Client Configuration (num_clients, clients_per_round)
  - Training Parameters (num_rounds, local_epochs, fl_method)
  - And more...
- **Real-time Updates** - All tabs update immediately when filters change
- **Clear Filters** - Reset to view all experiments

### Overview Tab
- **Summary Statistics Cards** - Total experiments, average accuracies (filtered)
- **Best/Worst Experiments** - Highlights top and bottom performers (filtered)
- **Final Accuracy Comparison Chart** - Bar chart comparing all experiments
- **Experiments Table** - Sortable table with all metrics and quick actions

### Comparison Tab
- **Experiment Selector** - Toggle experiments for comparison
- **Client Accuracy Over Rounds** - Line chart showing learning curves
- **Global Accuracy Over Rounds** - Line chart showing global model performance
- **Labeled Samples Progression** - Track active learning sample acquisition

### Details Tab (per experiment)
- **Key Metrics Cards** - Rounds, clients, best accuracies
- **Configuration Parameters** - Full display of experiment config
- **Baseline Comparison** - Visual improvement over baseline
- **Veracity Feedback Usage** - Stacked bar chart of veracity usage
- **Class Distributions** - Train/test class balance visualization
- **Experiment Path** - File system location for reference

## Output Format

The analysis JSON file has the following structure:

```json
{
  "summary": {
    "total_experiments": 10,
    "avg_final_client_acc": 85.42,
    "avg_final_global_acc": 87.31,
    "best_experiment": {
      "name": "exp_veracity_0.7",
      "final_acc": 92.15,
      "path": "/path/to/exp"
    },
    "avg_baseline_acc": 78.23,
    "avg_improvement_over_baseline": 7.19
  },
  "experiments": [
    {
      "exp_name": "experiment_1",
      "exp_path": "/path/to/experiment_1",
      "num_rounds": 10,
      "num_clients": 5,
      "final_avg_client_acc": 85.67,
      "final_global_test_acc": 87.42,
      "best_avg_client_acc": 86.21,
      "best_global_test_acc": 88.13,
      "baseline_acc": 78.45,
      "improvement_over_baseline": 7.22,
      "avg_labeled_samples": [100, 150, 200, ...],
      "avg_veracity_used": [0, 50, 100, ...],
      "avg_veracity_filtered": [0, 10, 15, ...],
      "rounds_data": [...],
      "train_class_distribution": {...},
      "test_class_distribution": {...},
      "per_client_distributions": {...},
      "config": {...}
    }
  ]
}
```

## Common Use Cases

### Compare Multiple Experiments

1. Run experiments with different configurations
2. Ensure all output to a common results directory
3. Run analysis script on the entire directory
4. Use dashboard's Comparison tab to visualize differences

### Filter by Configuration Parameters

1. Open dashboard with all experiments loaded
2. Use Filter Panel to select specific parameter values
   - Example: Select `dirichlet_alpha: 0.1` to see only highly heterogeneous experiments
   - Example: Select `veracity_threshold: 0.5` and `federated_mode: veracity_query`
3. Click "Apply Filters" to update all views
4. Compare filtered experiments in Comparison tab
5. Use "Clear All" to reset and try different filter combinations

### Analyze Veracity Impact

1. Filter by `federated_mode: veracity_query`
2. Further filter by specific `veracity_threshold` values
3. Compare accuracy trends across different thresholds
4. Check "Labeled Samples Progression" to see sample efficiency
5. Switch to standard mode and compare with filtered veracity experiments

### Study Data Heterogeneity Effects

1. Filter by `partition_mode: dirichlet`
2. Select multiple `dirichlet_alpha` values (e.g., 0.1, 0.5, 1.0)
3. Compare how heterogeneity affects convergence
4. Check if veracity query helps in highly heterogeneous settings
5. Compare with IID baseline by filtering `partition_mode: iid`

### Track Active Learning Performance

1. Load analysis in dashboard
2. Navigate to Comparison tab
3. Select experiments to compare
4. View "Labeled Samples Progression" chart
5. Correlate sample acquisition with accuracy improvements

### Identify Best Configuration

1. Run analysis script
2. Use filters to narrow down to specific setup (e.g., specific dataset, partition mode)
3. Open dashboard Overview tab
4. Sort experiments table by desired metric
5. Click "Details" on top performers for deeper analysis

### Baseline vs. Veracity Comparison

1. Ensure experiments include baseline runs
2. Run analysis script
3. Filter to compare specific configurations
4. Dashboard Overview shows average improvement
5. Use Comparison tab to see learning curves
6. Details tab shows per-experiment baseline comparison

### Multi-Dimensional Analysis

1. Filter by multiple parameters simultaneously
   - Example: `al_method: bayes_misp` + `partition_mode: dirichlet` + `dirichlet_alpha: 0.1`
2. Vary one parameter while keeping others fixed
3. Click "Apply Filters" between each combination
4. Build understanding of parameter interactions
5. Export findings from Details tab for each configuration

## Troubleshooting

### No experiments found
- Verify directory structure includes `global_metrics.json` files
- Check file permissions
- Try with `--verbose` flag to see parsing details

### Missing metrics in dashboard
- Some experiments may not have baseline or veracity data
- Dashboard handles missing data gracefully
- Optional fields show "-" when unavailable

### Dashboard not opening in browser
- Try manually opening the generated `.html` file
- Check that the HTML file was created successfully
- Use any modern web browser (Chrome, Firefox, Edge, Safari)

### JSON parsing errors
- Validate JSON output with a JSON validator
- Check for file corruption
- Re-run analysis script if needed

## Tips for Best Results

1. **Consistent Naming** - Use descriptive experiment names that reflect configurations
2. **Complete Runs** - Ensure experiments finish all rounds before analysis
3. **Baseline Comparison** - Always include baseline runs for context
4. **Multiple Seeds** - Run experiments with different random seeds for statistical significance
5. **Organized Structure** - Keep experiments in organized directory hierarchies

## Integration with Existing Workflow

This analysis toolkit is designed to work seamlessly with the existing federated learning pipeline:

```bash
# 1. Run federated experiments (existing workflow)
python train_federated_al.py --config config1.yaml

# 2. Run more experiments with different configs
python train_federated_al.py --config config2.yaml

# 3. Analyze all results
python analyze_federated_results.py ./experiments -o analysis.json

# 4. Generate and view dashboard
python view_federated_dashboard.py analysis.json

# Or generate HTML separately
python generate_html_dashboard.py analysis.json -o dashboard.html
```

## Requirements

- Python 3.7+
- NumPy (for data processing)
- Modern web browser (Chrome, Firefox, Edge, Safari)

No additional Python packages required! The HTML dashboard uses Chart.js loaded via CDN.

## Future Enhancements

Potential additions for future versions:
- Statistical significance tests
- Automatic hyperparameter optimization recommendations
- Export charts as images
- PDF report generation
- Real-time monitoring of running experiments
- Integration with experiment tracking tools (MLflow, Weights & Biases)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your experiment outputs match the expected format
3. Review the FederatedServer code for output format details
