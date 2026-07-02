# Experiment Filtering Guide

The dashboard now includes powerful filtering capabilities to help you analyze specific subsets of experiments based on their configuration parameters.

## How Filtering Works

### 1. Automatic Parameter Detection

The analysis script automatically extracts all configuration parameters from your experiments. When you run:

```bash
python analyze_federated_results.py experiments/ -o results.json
```

It scans all `config.json` files and identifies unique values for each parameter. These become available as filters in the dashboard.

### 2. Filter Panel

The dashboard displays a filter panel at the top with dropdown menus for each detected parameter:

- **Dataset** - cifar10, cifar100, mnist
- **AL Method** - random, bayes_misp, etc.
- **Partition Mode** - iid, dirichlet
- **Dirichlet Alpha** - 0.1, 0.5, 1.0, etc.
- **Federated Mode** - standard, veracity_query
- **Veracity Threshold** - 0.25, 0.5, 0.75
- **Veracity Loss Weight** - 1, 2, 4, 8
- **Queries Per Round** - 0, 1, 2, 100
- **Clients Per Round** - 3, 5, 8, 10
- **Num Rounds** - 5, 10, 15, 20, 50
- **Local Epochs** - 1, 2, 3, 5, 10, 200
- **FL Method** - fedavg, fedprox
- **FedProx Mu** - 0.001, 0.01, 0.1
- **Client Labels Initial Size** - 5, 10, 20, 50, 100, 200
- And more...

### 3. Multi-Select Filtering

Each dropdown supports multi-select (hold Ctrl/Cmd to select multiple values):

**Example 1: Compare different Dirichlet alpha values**
```
Filter: dirichlet_alpha = [0.1, 0.5, 1.0]
Result: Shows only experiments with these alpha values
```

**Example 2: Analyze veracity impact**
```
Filter: federated_mode = [veracity_query]
        veracity_threshold = [0.5]
Result: Shows only veracity experiments with 0.5 threshold
```

**Example 3: Compare IID vs Non-IID**
```
Filter: partition_mode = [iid, dirichlet]
        dirichlet_alpha = [0.1]  (for non-IID)
Result: Shows IID experiments and highly heterogeneous non-IID
```

### 4. Apply and Clear

- **Apply Filters** - Updates all tabs with filtered experiments
- **Clear All** - Resets to show all experiments

### 5. Real-Time Updates

When you apply filters:
- **Overview Tab** - Updates summary stats, charts, and table
- **Comparison Tab** - Updates experiment selector and all charts
- **Details Tab** - Updates dropdown with filtered experiments

## Common Filtering Patterns

### Pattern 1: Isolate Single Variable

**Goal:** Understand impact of veracity threshold

```
Steps:
1. Set all parameters to single values EXCEPT veracity_threshold
2. Select multiple veracity_threshold values [0.25, 0.5, 0.75]
3. Apply filters
4. Compare in Comparison tab
```

### Pattern 2: Category Comparison

**Goal:** Compare IID vs Non-IID with different heterogeneity levels

```
Steps:
1. Filter: partition_mode = [iid, dirichlet]
2. Filter: dirichlet_alpha = [0.1, 0.5, 1.0]
3. Apply filters
4. View accuracy trends in Comparison tab
```

### Pattern 3: Best Configuration Search

**Goal:** Find best performing setup for specific scenario

```
Steps:
1. Filter: dataset = [cifar100]
2. Filter: federated_mode = [veracity_query]
3. Apply filters
4. Sort by accuracy in Overview tab table
5. Click Details on top performer
6. Note configuration parameters
```

### Pattern 4: Parameter Interaction Study

**Goal:** Understand how veracity threshold and loss weight interact

```
Steps:
1. Filter: veracity_threshold = [0.5]
2. Filter: veracity_loss_weight = [1, 2, 4, 8]
3. Keep other params fixed
4. Apply filters and compare
5. Repeat with different thresholds
```

### Pattern 5: Baseline Comparison

**Goal:** Compare veracity vs standard mode

```
Steps:
1. Filter: federated_mode = [standard, veracity_query]
2. Keep dataset and partition settings same
3. Apply filters
4. Check improvement in Overview tab
```

## Advanced Filtering Techniques

### Technique 1: Progressive Refinement

Start broad and narrow down:

```
Round 1: Filter dataset = [cifar100]
         Result: 50 experiments

Round 2: Add partition_mode = [dirichlet]
         Result: 30 experiments

Round 3: Add dirichlet_alpha = [0.1]
         Result: 10 experiments

Round 4: Add federated_mode = [veracity_query]
         Result: 5 experiments
```

### Technique 2: A/B Testing

Compare two configurations side-by-side:

```
Setup A:
- al_method = [bayes_misp]
- partition_mode = [dirichlet]
- dirichlet_alpha = [0.1]
- federated_mode = [standard]

Setup B:
- al_method = [bayes_misp]
- partition_mode = [dirichlet]
- dirichlet_alpha = [0.1]
- federated_mode = [veracity_query]
- veracity_threshold = [0.5]

View both in Comparison tab
```

### Technique 3: Sensitivity Analysis

Test sensitivity to a single parameter:

```
Fix all parameters except one:
- dataset = [cifar100]
- partition_mode = [dirichlet]
- dirichlet_alpha = [0.1]
- federated_mode = [veracity_query]
- queries_per_round = [100]
- veracity_loss_weight = [2]
- veracity_threshold = [0.25, 0.5, 0.75]  ← VARY THIS

Apply and observe impact on accuracy
```

## Configuration File Format

For filtering to work, each experiment must have a `config.json` file in its directory:

```json
{
  "dataset": "cifar100",
  "al_method": "bayes_misp",
  "partition_mode": "dirichlet",
  "dirichlet_alpha": "0.5",
  "federated_mode": "veracity_query",
  "queries_per_round": 100,
  "veracity_threshold": "0.5",
  "veracity_loss_weight": "2",
  "num_clients": 10,
  "num_rounds": 5,
  "clients_per_round": 10,
  "local_epochs": 200,
  "fl_method": "fedavg",
  "client_labels_initial_size": 100
}
```

This file is automatically created by `train_federated_al.py` when you run experiments.

## Tips for Effective Filtering

1. **Start with few filters** - Add more constraints progressively
2. **Use multi-select strategically** - Select related values together
3. **Check experiment count** - Badge shows how many experiments match filters
4. **Clear and reapply** - Don't hesitate to reset and try different combinations
5. **Document findings** - Note which filter combinations yield insights
6. **Export configs** - Use Details tab to see full config of interesting experiments

## Troubleshooting

### No experiments match filters
- Clear all filters and reapply one at a time
- Check that config.json files exist in experiment directories
- Verify parameter values match exactly (case-sensitive)

### Filter dropdown is empty
- No experiments have this parameter in their config
- Parameter may have been added in later experiments only
- Re-run analysis script to update filter values

### Filters not updating charts
- Click "Apply Filters" button after making selections
- Check browser console for any JavaScript errors
- Try refreshing the page and reloading the JSON

### Missing parameters in Details tab
- Some experiments may not have all parameters
- Older experiments might use different parameter names
- Parameters with null/None values are excluded

## Example Workflow

**Research Question:** Does increasing veracity threshold improve accuracy in heterogeneous settings?

```bash
# 1. Generate data with your experiments
python analyze_federated_results.py experiments/ -o results.json
python view_federated_dashboard.py results.json

# 2. In dashboard:
#    - Filter: partition_mode = [dirichlet]
#    - Filter: dirichlet_alpha = [0.1]  # Highly heterogeneous
#    - Filter: federated_mode = [veracity_query]
#    - Filter: veracity_threshold = [0.25, 0.5, 0.75]
#    - Apply Filters

# 3. Go to Comparison tab:
#    - Select all experiments
#    - Observe accuracy trends over rounds
#    - Check if higher thresholds converge better

# 4. Go to Overview tab:
#    - Sort table by final accuracy
#    - Identify best threshold

# 5. Go to Details tab:
#    - Select best experiment
#    - View full config
#    - Note: veracity_threshold = 0.5 performs best
```

## Conclusion

Filtering transforms the dashboard from a viewing tool into an analytical powerhouse. By systematically filtering and comparing experiments, you can:

- Identify optimal hyperparameters
- Understand parameter interactions
- Compare algorithm variants
- Validate research hypotheses
- Communicate findings effectively

Use filters to ask specific questions of your data, and let the dashboard reveal the answers!
