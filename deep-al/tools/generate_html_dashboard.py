"""
Generate a standalone HTML dashboard from federated analysis results.
This creates a single HTML file that can be opened in any web browser.

Usage:
    python generate_html_dashboard.py <analysis_json> [--output dashboard.html]
"""

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated Learning Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #f3f4f6;
            color: #1f2937;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .header {
            background: white;
            padding: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #6b7280;
        }
        
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .tab-button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            background: #e5e7eb;
            color: #374151;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .tab-button:hover {
            background: #d1d5db;
        }
        
        .tab-button.active {
            background: #2563eb;
            color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid;
        }
        
        .stat-card.blue { border-color: #2563eb; }
        .stat-card.green { border-color: #10b981; }
        .stat-card.purple { border-color: #7c3aed; }
        .stat-card.amber { border-color: #f59e0b; }
        
        .stat-card h3 {
            font-size: 0.875rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            font-weight: 600;
        }
        
        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #1f2937;
        }
        
        .stat-card .label {
            font-size: 0.875rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }
        
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .chart-container h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        
        .chart-wrapper {
            position: relative;
            height: 400px;
        }
        
        table {
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 0.5rem;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        thead {
            background: #f9fafb;
        }
        
        th {
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #6b7280;
            border-bottom: 1px solid #e5e7eb;
        }
        
        td {
            padding: 1rem;
            border-bottom: 1px solid #e5e7eb;
            font-size: 0.875rem;
        }
        
        tbody tr:hover {
            background: #f9fafb;
        }
        
        .positive {
            color: #10b981;
            font-weight: 600;
        }
        
        .negative {
            color: #ef4444;
            font-weight: 600;
        }
        
        .experiment-selector {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .experiment-selector h3 {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        
        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        
        .checkbox-label {
            display: flex;
            align-items: center;
            padding: 0.5rem 0.75rem;
            background: #f3f4f6;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .checkbox-label:hover {
            background: #e5e7eb;
        }
        
        .checkbox-label input {
            margin-right: 0.5rem;
        }
        
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .detail-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .detail-card h3 {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        
        .detail-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .detail-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            background: #f9fafb;
            border-radius: 0.25rem;
        }
        
        .detail-item .key {
            color: #6b7280;
            font-size: 0.875rem;
        }
        
        .detail-item .value {
            font-weight: 600;
            font-size: 0.875rem;
        }
        
        .filter-panel {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .filter-panel h3 {
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        
        .filter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .filter-item {
            display: flex;
            flex-direction: column;
        }
        
        .filter-item label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }
        
        .filter-item select {
            padding: 0.5rem;
            border: 1px solid #d1d5db;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            background: white;
            cursor: pointer;
        }
        
        .filter-item select:focus {
            outline: none;
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        .filter-actions {
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
        }
        
        .filter-button {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .filter-button.primary {
            background: #2563eb;
            color: white;
        }
        
        .filter-button.primary:hover {
            background: #1d4ed8;
        }
        
        .filter-button.secondary {
            background: #e5e7eb;
            color: #374151;
        }
        
        .filter-button.secondary:hover {
            background: #d1d5db;
        }
        
        .filter-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            background: #dbeafe;
            color: #1e40af;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 500;
            margin-left: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Federated Learning Dashboard</h1>
            <p>Analysis of federated learning experiment results</p>
            
            <div class="tabs">
                <button class="tab-button active" onclick="switchTab('overview')">Overview</button>
                <button class="tab-button" onclick="switchTab('comparison')">Comparison</button>
                <button class="tab-button" onclick="switchTab('details')">Details</button>
            </div>
        </div>
        
        <!-- Filter Panel -->
        <div class="filter-panel" id="filter-panel">
            <h3>Filter Experiments <span class="filter-badge" id="filter-count">0 active filters</span></h3>
            <div class="filter-grid" id="filter-grid"></div>
            <div class="filter-actions">
                <button class="filter-button primary" onclick="applyFilters()">Apply Filters</button>
                <button class="filter-button secondary" onclick="clearFilters()">Clear All</button>
            </div>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview-tab" class="tab-content active">
            <div class="stats-grid">
                <div class="stat-card blue">
                    <h3>Total Experiments</h3>
                    <div class="value" id="stat-total-exp">0</div>
                </div>
                <div class="stat-card green">
                    <h3>Avg Final Client Acc</h3>
                    <div class="value" id="stat-avg-client">0%</div>
                </div>
                <div class="stat-card purple">
                    <h3>Avg Final Global Acc</h3>
                    <div class="value" id="stat-avg-global">0%</div>
                </div>
                <div class="stat-card amber" id="baseline-card" style="display: none;">
                    <h3>Avg Improvement</h3>
                    <div class="value" id="stat-improvement">0%</div>
                    <div class="label" id="stat-baseline-label">over baseline</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Final Accuracy Comparison</h2>
                <div class="chart-wrapper">
                    <canvas id="overview-chart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Experiments Summary</h2>
                <table id="experiments-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Rounds</th>
                            <th>Clients</th>
                            <th>Final Client Acc</th>
                            <th>Final Global Acc</th>
                            <th>Baseline</th>
                            <th>Improvement</th>
                        </tr>
                    </thead>
                    <tbody id="experiments-tbody">
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Comparison Tab -->
        <div id="comparison-tab" class="tab-content">
            <div class="experiment-selector">
                <h3>Select Experiments to Compare</h3>
                <div class="checkbox-group" id="experiment-checkboxes"></div>
            </div>
            
            <div class="chart-container">
                <h2>Client Accuracy Over Rounds</h2>
                <div class="chart-wrapper">
                    <canvas id="client-acc-chart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Global Test Accuracy Over Rounds</h2>
                <div class="chart-wrapper">
                    <canvas id="global-acc-chart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Labeled Samples Over Rounds</h2>
                <div class="chart-wrapper">
                    <canvas id="labeled-samples-chart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Details Tab -->
        <div id="details-tab" class="tab-content">
            <div class="experiment-selector">
                <h3>Select Experiment</h3>
                <select id="detail-experiment-select" onchange="updateDetails()" style="width: 100%; padding: 0.5rem; border-radius: 0.375rem; border: 1px solid #d1d5db;">
                    <option value="">Choose an experiment...</option>
                </select>
            </div>
            
            <div id="detail-content"></div>
        </div>
    </div>
    
    <script>
        // Data will be injected here
        const data = DATA_PLACEHOLDER;
        
        let charts = {};
        let selectedExperiments = [];
        let activeFilters = {};
        let filteredData = { ...data };
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeFilters();
            applyFilters(); // This will initialize everything with potentially filtered data
        });
        
        function initializeFilters() {
            const filterGrid = document.getElementById('filter-grid');
            const filterValues = data.summary.filter_values || {};
            
            // Create filter dropdowns for each parameter
            Object.entries(filterValues).forEach(([param, values]) => {
                if (values.length > 0) {
                    const filterItem = document.createElement('div');
                    filterItem.className = 'filter-item';
                    
                    const label = document.createElement('label');
                    label.textContent = param.replace(/_/g, ' ');
                    
                    const select = document.createElement('select');
                    select.id = 'filter-' + param;
                    select.multiple = true;
                    select.size = Math.min(values.length + 1, 5);
                    
                    // Add "All" option
                    const allOption = document.createElement('option');
                    allOption.value = '';
                    allOption.textContent = '(All)';
                    allOption.selected = true;
                    select.appendChild(allOption);
                    
                    // Add value options
                    values.forEach(val => {
                        const option = document.createElement('option');
                        option.value = val;
                        option.textContent = val;
                        select.appendChild(option);
                    });
                    
                    filterItem.appendChild(label);
                    filterItem.appendChild(select);
                    filterGrid.appendChild(filterItem);
                }
            });
        }
        
        function applyFilters() {
            // Collect active filters
            activeFilters = {};
            let filterCount = 0;
            
            const filterValues = data.summary.filter_values || {};
            Object.keys(filterValues).forEach(param => {
                const select = document.getElementById('filter-' + param);
                if (select) {
                    const selected = Array.from(select.selectedOptions)
                        .map(opt => opt.value)
                        .filter(val => val !== '');
                    
                    if (selected.length > 0) {
                        activeFilters[param] = selected;
                        filterCount++;
                    }
                }
            });
            
            // Update filter count badge
            const badge = document.getElementById('filter-count');
            badge.textContent = filterCount === 0 ? 'No active filters' : 
                                filterCount === 1 ? '1 active filter' :
                                filterCount + ' active filters';
            
            // Filter experiments
            filteredData.experiments = data.experiments.filter(exp => {
                // Check if experiment matches all active filters
                for (const [param, values] of Object.entries(activeFilters)) {
                    const expValue = exp.config && exp.config[param] ? String(exp.config[param]) : null;
                    if (!expValue || !values.includes(expValue)) {
                        return false;
                    }
                }
                return true;
            });
            
            // Update summary stats for filtered data
            if (filteredData.experiments.length > 0) {
                filteredData.summary = {
                    ...data.summary,
                    total_experiments: filteredData.experiments.length,
                    avg_final_client_acc: filteredData.experiments.reduce((sum, e) => sum + e.final_avg_client_acc, 0) / filteredData.experiments.length,
                    avg_final_global_acc: filteredData.experiments.reduce((sum, e) => sum + e.final_global_test_acc, 0) / filteredData.experiments.length
                };
                
                // Recalculate best/worst
                const sorted = [...filteredData.experiments].sort((a, b) => b.final_avg_client_acc - a.final_avg_client_acc);
                filteredData.summary.best_experiment = {
                    name: sorted[0].exp_name,
                    final_acc: sorted[0].final_avg_client_acc,
                    path: sorted[0].exp_path
                };
                filteredData.summary.worst_experiment = {
                    name: sorted[sorted.length - 1].exp_name,
                    final_acc: sorted[sorted.length - 1].final_avg_client_acc,
                    path: sorted[sorted.length - 1].exp_path
                };
            }
            
            // Reset selected experiments for comparison
            selectedExperiments = filteredData.experiments.map(e => e.exp_name);
            
            // Reinitialize all views with filtered data
            initializeOverview();
            initializeComparison();
            initializeDetails();
        }
        
        function clearFilters() {
            // Clear all filter selections
            const filterValues = data.summary.filter_values || {};
            Object.keys(filterValues).forEach(param => {
                const select = document.getElementById('filter-' + param);
                if (select) {
                    Array.from(select.options).forEach(opt => {
                        opt.selected = (opt.value === '');
                    });
                }
            });
            
            applyFilters();
        }
        
        function switchTab(tab) {
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
            
            if (tab === 'comparison') {
                updateComparisonCharts();
            }
        }
        
        function initializeOverview() {
            // Update stats
            document.getElementById('stat-total-exp').textContent = filteredData.summary.total_experiments;
            document.getElementById('stat-avg-client').textContent = filteredData.summary.avg_final_client_acc.toFixed(2) + '%';
            document.getElementById('stat-avg-global').textContent = filteredData.summary.avg_final_global_acc.toFixed(2) + '%';
            
            if (filteredData.summary.avg_improvement_over_baseline) {
                document.getElementById('baseline-card').style.display = 'block';
                document.getElementById('stat-improvement').textContent = '+' + filteredData.summary.avg_improvement_over_baseline.toFixed(2) + '%';
            } else {
                document.getElementById('baseline-card').style.display = 'none';
            }
            
            // Destroy existing chart if it exists
            if (charts.overview) {
                charts.overview.destroy();
            }
            
            // Create overview chart
            const ctx = document.getElementById('overview-chart').getContext('2d');
            const chartData = {
                labels: filteredData.experiments.map(e => e.exp_name.length > 20 ? e.exp_name.substring(0, 17) + '...' : e.exp_name),
                datasets: [
                    {
                        label: 'Client Acc',
                        data: filteredData.experiments.map(e => e.final_avg_client_acc),
                        backgroundColor: '#2563eb'
                    },
                    {
                        label: 'Global Acc',
                        data: filteredData.experiments.map(e => e.final_global_test_acc),
                        backgroundColor: '#7c3aed'
                    }
                ]
            };
            
            if (filteredData.experiments.some(e => e.baseline_acc)) {
                chartData.datasets.push({
                    label: 'Baseline',
                    data: filteredData.experiments.map(e => e.baseline_acc || 0),
                    backgroundColor: '#94a3b8'
                });
            }
            
            charts.overview = new Chart(ctx, {
                type: 'bar',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: { display: true, text: 'Accuracy (%)' }
                        }
                    }
                }
            });
            
            // Populate table
            const tbody = document.getElementById('experiments-tbody');
            tbody.innerHTML = ''; // Clear existing rows
            filteredData.experiments.forEach(exp => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${exp.exp_name}</td>
                    <td>${exp.num_rounds}</td>
                    <td>${exp.num_clients}</td>
                    <td>${exp.final_avg_client_acc.toFixed(2)}%</td>
                    <td>${exp.final_global_test_acc.toFixed(2)}%</td>
                    <td>${exp.baseline_acc ? exp.baseline_acc.toFixed(2) + '%' : '-'}</td>
                    <td class="${exp.improvement_over_baseline > 0 ? 'positive' : 'negative'}">
                        ${exp.improvement_over_baseline ? (exp.improvement_over_baseline > 0 ? '+' : '') + exp.improvement_over_baseline.toFixed(2) + '%' : '-'}
                    </td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function initializeComparison() {
            // Create checkboxes
            const container = document.getElementById('experiment-checkboxes');
            container.innerHTML = ''; // Clear existing checkboxes
            filteredData.experiments.forEach(exp => {
                const label = document.createElement('label');
                label.className = 'checkbox-label';
                label.innerHTML = `
                    <input type="checkbox" checked onchange="toggleExperiment('${exp.exp_name}')" id="cb-${exp.exp_name}">
                    ${exp.exp_name}
                `;
                container.appendChild(label);
            });
            
            // Initialize charts
            initComparisonCharts();
        }
        
        function initComparisonCharts() {
            // Destroy existing charts if they exist
            if (charts.clientAcc) charts.clientAcc.destroy();
            if (charts.globalAcc) charts.globalAcc.destroy();
            if (charts.labeledSamples) charts.labeledSamples.destroy();
            
            const ctx1 = document.getElementById('client-acc-chart').getContext('2d');
            charts.clientAcc = new Chart(ctx1, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { title: { display: true, text: 'Accuracy (%)' } },
                        x: { title: { display: true, text: 'Round' } }
                    }
                }
            });
            
            const ctx2 = document.getElementById('global-acc-chart').getContext('2d');
            charts.globalAcc = new Chart(ctx2, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { title: { display: true, text: 'Accuracy (%)' } },
                        x: { title: { display: true, text: 'Round' } }
                    }
                }
            });
            
            const ctx3 = document.getElementById('labeled-samples-chart').getContext('2d');
            charts.labeledSamples = new Chart(ctx3, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { title: { display: true, text: 'Labeled Samples' } },
                        x: { title: { display: true, text: 'Round' } }
                    }
                }
            });
            
            updateComparisonCharts();
        }
        
        function toggleExperiment(expName) {
            const checkbox = document.getElementById('cb-' + expName);
            if (checkbox.checked) {
                selectedExperiments.push(expName);
            } else {
                selectedExperiments = selectedExperiments.filter(n => n !== expName);
            }
            updateComparisonCharts();
        }
        
        function updateComparisonCharts() {
            const colors = ['#2563eb', '#7c3aed', '#db2777', '#dc2626', '#ea580c', '#ca8a04', '#16a34a', '#0891b2', '#4f46e5', '#be123c'];
            const selectedExps = filteredData.experiments.filter(e => selectedExperiments.includes(e.exp_name));
            
            if (selectedExps.length === 0) return;
            
            const maxRounds = Math.max(...selectedExps.map(e => e.num_rounds));
            const rounds = Array.from({length: maxRounds}, (_, i) => i);
            
            // Client accuracy chart
            charts.clientAcc.data.labels = rounds;
            charts.clientAcc.data.datasets = selectedExps.map((exp, idx) => ({
                label: exp.exp_name,
                data: exp.rounds_data.map(r => r.avg_client_acc),
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length] + '20',
                tension: 0.1
            }));
            charts.clientAcc.update();
            
            // Global accuracy chart
            charts.globalAcc.data.labels = rounds;
            charts.globalAcc.data.datasets = selectedExps.map((exp, idx) => ({
                label: exp.exp_name,
                data: exp.rounds_data.map(r => r.global_test_acc),
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length] + '20',
                tension: 0.1
            }));
            charts.globalAcc.update();
            
            // Labeled samples chart
            charts.labeledSamples.data.labels = rounds;
            charts.labeledSamples.data.datasets = selectedExps.map((exp, idx) => ({
                label: exp.exp_name,
                data: exp.avg_labeled_samples,
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length] + '20',
                tension: 0.1
            }));
            charts.labeledSamples.update();
        }
        
        function initializeDetails() {
            const select = document.getElementById('detail-experiment-select');
            select.innerHTML = '<option value="">Choose an experiment...</option>'; // Clear existing options
            filteredData.experiments.forEach(exp => {
                const option = document.createElement('option');
                option.value = exp.exp_name;
                option.textContent = exp.exp_name;
                select.appendChild(option);
            });
        }
        
        function updateDetails() {
            const expName = document.getElementById('detail-experiment-select').value;
            if (!expName) return;
            
            const exp = filteredData.experiments.find(e => e.exp_name === expName);
            if (!exp) return;
            
            const content = document.getElementById('detail-content');
            let html = `
                <div class="stats-grid">
                    <div class="stat-card blue">
                        <h3>Rounds</h3>
                        <div class="value">${exp.num_rounds}</div>
                    </div>
                    <div class="stat-card purple">
                        <h3>Clients</h3>
                        <div class="value">${exp.num_clients}</div>
                    </div>
                    <div class="stat-card green">
                        <h3>Best Client Acc</h3>
                        <div class="value">${exp.best_avg_client_acc.toFixed(2)}%</div>
                    </div>
                    <div class="stat-card amber">
                        <h3>Best Global Acc</h3>
                        <div class="value">${exp.best_global_test_acc.toFixed(2)}%</div>
                    </div>
                </div>
            `;
            
            if (exp.baseline_acc) {
                html += `
                    <div class="chart-container">
                        <h2>Baseline Comparison</h2>
                        <div class="detail-list">
                            <div class="detail-item">
                                <span class="key">Baseline Accuracy</span>
                                <span class="value">${exp.baseline_acc.toFixed(2)}%</span>
                            </div>
                            <div class="detail-item">
                                <span class="key">Final Accuracy</span>
                                <span class="value">${exp.final_avg_client_acc.toFixed(2)}%</span>
                            </div>
                            <div class="detail-item">
                                <span class="key">Improvement</span>
                                <span class="value positive">+${exp.improvement_over_baseline.toFixed(2)}%</span>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            if (exp.train_class_distribution) {
                html += `
                    <div class="detail-grid">
                        <div class="detail-card">
                            <h3>Train Class Distribution</h3>
                            <div class="detail-list">
                                ${Object.entries(exp.train_class_distribution).map(([cls, count]) => `
                                    <div class="detail-item">
                                        <span class="key">Class ${cls}</span>
                                        <span class="value">${count}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ${exp.test_class_distribution ? `
                            <div class="detail-card">
                                <h3>Test Class Distribution</h3>
                                <div class="detail-list">
                                    ${Object.entries(exp.test_class_distribution).map(([cls, count]) => `
                                        <div class="detail-item">
                                            <span class="key">Class ${cls}</span>
                                            <span class="value">${count}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `;
            }
            
            // Add config parameters if available
            if (exp.config && Object.keys(exp.config).length > 0) {
                html += `
                    <div class="chart-container">
                        <h2>Configuration Parameters</h2>
                        <div class="detail-list">
                `;
                
                Object.entries(exp.config).sort().forEach(([key, value]) => {
                    if (value !== null && value !== undefined) {
                        html += `
                            <div class="detail-item">
                                <span class="key">${key.replace(/_/g, ' ')}</span>
                                <span class="value">${value}</span>
                            </div>
                        `;
                    }
                });
                
                html += `
                        </div>
                    </div>
                `;
            }
            
            html += `
                <div class="chart-container">
                    <h2>Experiment Path</h2>
                    <code style="padding: 1rem; background: #f3f4f6; border-radius: 0.375rem; display: block; overflow-x: auto;">
                        ${exp.exp_path}
                    </code>
                </div>
            `;
            
            content.innerHTML = html;
        }
    </script>
</body>
</html>
"""


def generate_html_dashboard(analysis_json_path: str, output_html_path: str):
    """Generate HTML dashboard from analysis JSON."""
    # Load analysis data
    with open(analysis_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Generate HTML with embedded data
    html = HTML_TEMPLATE.replace('DATA_PLACEHOLDER', json.dumps(data))
    
    # Write output
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard from federated analysis results"
    )
    parser.add_argument(
        "analysis_json",
        type=str,
        help="Path to the analysis JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="federated_dashboard.html",
        help="Output HTML file path (default: federated_dashboard.html)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HTML Dashboard Generator")
    print("=" * 60)
    
    json_path = Path(args.analysis_json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {args.analysis_json}")
        return 1
    
    print(f"\nGenerating dashboard from: {json_path}")
    
    output_path = generate_html_dashboard(str(json_path), args.output)
    
    print(f"✓ Dashboard generated: {output_path.absolute()}")
    print("\nTo view the dashboard:")
    print(f"  1. Open in browser: {output_path.absolute()}")
    print(f"  2. Or run: start {output_path.name}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
