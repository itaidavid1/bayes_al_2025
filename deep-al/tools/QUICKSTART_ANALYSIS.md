# Quick Start Guide - Federated Learning Analysis

## 1. Test the Tools (Optional but Recommended)

First, verify everything works with sample data:

```bash
cd c:\Users\User\Lab\py_repos\TypiClust\deep-al\tools
python test_analysis_tools.py
```

This creates sample experiments and tests the analysis pipeline.

## 2. Analyze Your Real Experiments

Point the script to your results directory:

```bash
python analyze_federated_results.py c:\path\to\your\experiments -o my_results.json
```

**Example with actual path:**
```bash
python analyze_federated_results.py c:\Users\User\Lab\py_repos\TypiClust\deep-al\experiments -o federated_analysis.json
```

## 3. View the Dashboard

Generate an HTML dashboard and open it in your browser:

```bash
python generate_html_dashboard.py my_results.json -o dashboard.html
start dashboard.html
```

The HTML dashboard works in any web browser and includes:
- Interactive charts with Chart.js
- Overview, Comparison, and Details tabs
- All metrics visualized beautifully

## What You'll See

### Filter Panel (Available on all tabs)
- Dynamic filters based on your experiment configs
- Multi-select filtering (e.g., filter by veracity_threshold, dirichlet_alpha)
- Apply/Clear filters to narrow down experiments
- Real-time updates across all tabs

### Overview Tab
- Summary cards with key metrics
- Best/worst experiments
- Bar chart comparing all experiments
- Sortable table with all data

### Comparison Tab
- Select experiments to compare
- Line charts showing accuracy over rounds
- Active learning progression
- Veracity usage patterns

### Details Tab (click on any experiment)
- Deep dive into single experiment
- Full configuration parameters display
- Baseline comparison
- Class distributions
- Round-by-round metrics

## Example Workflow

```bash
# 1. Analyze your experiments
python analyze_federated_results.py experiments/ -o results.json

# 2. Generate and view dashboard
python view_federated_dashboard.py results.json

# 3. In the dashboard:
#    - Use filters to select dirichlet_alpha = 0.1
#    - Compare different veracity_threshold values
#    - Check which configuration performs best
#    - Export findings from Details tab
```

## Common Commands

```bash
# Analyze with verbose output
python analyze_federated_results.py experiments/ -o results.json -v

# Generate HTML dashboard
python generate_html_dashboard.py results.json -o dashboard.html

# Generate and open dashboard automatically
python view_federated_dashboard.py results.json

# Test the tools
python test_analysis_tools.py
```

## File Locations

All tools are in: `c:\Users\User\Lab\py_repos\TypiClust\deep-al\tools\`

- `analyze_federated_results.py` - Main analysis script
- `generate_html_dashboard.py` - HTML dashboard generator
- `view_federated_dashboard.py` - Dashboard launcher (generates + opens HTML)
- `test_analysis_tools.py` - Testing script
- `README_FEDERATED_ANALYSIS.md` - Full documentation
- `QUICKSTART_ANALYSIS.md` - This file

## Troubleshooting

**No experiments found?**
- Check that your directory contains folders with `global_metrics.json` files
- Run with `-v` flag to see what's being scanned

**Dashboard not opening?**
- Check that the HTML file was generated successfully
- Try opening the `.html` file manually in your browser
- Works in any modern browser (Chrome, Firefox, Edge, Safari)

**Missing metrics?**
- Some experiments might not have baseline data - that's OK
- Optional fields show as "-" in the dashboard

## Need Help?

See the full documentation: `README_FEDERATED_ANALYSIS.md`
