import React, { useState, useMemo } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';

interface ExperimentMetrics {
  exp_name: string;
  exp_path: string;
  num_rounds: number;
  num_clients: number;
  final_avg_client_acc: number;
  final_global_test_acc: number;
  best_avg_client_acc: number;
  best_global_test_acc: number;
  baseline_acc?: number;
  improvement_over_baseline?: number;
  avg_labeled_samples: number[];
  avg_veracity_used: number[];
  avg_veracity_filtered: number[];
  rounds_data: any[];
  train_class_distribution?: Record<string, number>;
  test_class_distribution?: Record<string, number>;
  per_client_distributions?: Record<string, Record<string, number>>;
  config?: any;
}

interface DashboardData {
  summary: any;
  experiments: ExperimentMetrics[];
}

// Sample data structure - replace with actual loaded data
const defaultData: DashboardData = {
  summary: {
    total_experiments: 0,
    avg_final_client_acc: 0,
    avg_final_global_acc: 0
  },
  experiments: []
};

const COLORS = [
  '#2563eb', '#7c3aed', '#db2777', '#dc2626', '#ea580c',
  '#ca8a04', '#16a34a', '#0891b2', '#4f46e5', '#be123c'
];

export default function FederatedDashboard() {
  // In production, load from JSON file passed as prop
  const [data, setData] = useState<DashboardData>(defaultData);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'overview' | 'comparison' | 'details'>('overview');
  const [selectedExpDetail, setSelectedExpDetail] = useState<string | null>(null);

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const jsonData = JSON.parse(e.target?.result as string);
          setData(jsonData);
          // Auto-select all experiments for comparison
          setSelectedExperiments(jsonData.experiments.map((exp: ExperimentMetrics) => exp.exp_name));
        } catch (error) {
          alert('Error parsing JSON file: ' + error);
        }
      };
      reader.readAsText(file);
    }
  };

  // Toggle experiment selection for comparison
  const toggleExperiment = (expName: string) => {
    setSelectedExperiments(prev =>
      prev.includes(expName)
        ? prev.filter(name => name !== expName)
        : [...prev, expName]
    );
  };

  // Get selected experiment objects
  const selectedExps = useMemo(
    () => data.experiments.filter(exp => selectedExperiments.includes(exp.exp_name)),
    [data.experiments, selectedExperiments]
  );

  // Prepare data for accuracy comparison chart
  const accuracyComparisonData = useMemo(() => {
    if (selectedExps.length === 0) return [];
    
    const maxRounds = Math.max(...selectedExps.map(exp => exp.num_rounds));
    const chartData = [];
    
    for (let round = 0; round < maxRounds; round++) {
      const dataPoint: any = { round };
      selectedExps.forEach(exp => {
        if (round < exp.rounds_data.length) {
          dataPoint[exp.exp_name] = exp.rounds_data[round].avg_client_acc;
        }
      });
      chartData.push(dataPoint);
    }
    
    return chartData;
  }, [selectedExps]);

  // Prepare data for global test accuracy chart
  const globalAccuracyData = useMemo(() => {
    if (selectedExps.length === 0) return [];
    
    const maxRounds = Math.max(...selectedExps.map(exp => exp.num_rounds));
    const chartData = [];
    
    for (let round = 0; round < maxRounds; round++) {
      const dataPoint: any = { round };
      selectedExps.forEach(exp => {
        if (round < exp.rounds_data.length) {
          dataPoint[exp.exp_name] = exp.rounds_data[round].global_test_acc;
        }
      });
      chartData.push(dataPoint);
    }
    
    return chartData;
  }, [selectedExps]);

  // Prepare data for labeled samples chart
  const labeledSamplesData = useMemo(() => {
    if (selectedExps.length === 0) return [];
    
    const maxRounds = Math.max(...selectedExps.map(exp => exp.num_rounds));
    const chartData = [];
    
    for (let round = 0; round < maxRounds; round++) {
      const dataPoint: any = { round };
      selectedExps.forEach(exp => {
        if (round < exp.avg_labeled_samples.length) {
          dataPoint[exp.exp_name] = exp.avg_labeled_samples[round];
        }
      });
      chartData.push(dataPoint);
    }
    
    return chartData;
  }, [selectedExps]);

  // Prepare final comparison data
  const finalComparisonData = useMemo(() => {
    return data.experiments.map(exp => ({
      name: exp.exp_name.length > 20 ? exp.exp_name.substring(0, 17) + '...' : exp.exp_name,
      'Client Acc': exp.final_avg_client_acc,
      'Global Acc': exp.final_global_test_acc,
      'Baseline': exp.baseline_acc || 0,
      fullName: exp.exp_name
    }));
  }, [data.experiments]);

  // Overview section
  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
          <h3 className="text-sm font-medium text-blue-600 mb-2">Total Experiments</h3>
          <p className="text-3xl font-bold text-blue-900">{data.summary.total_experiments}</p>
        </div>
        
        <div className="bg-green-50 p-6 rounded-lg border border-green-200">
          <h3 className="text-sm font-medium text-green-600 mb-2">Avg Final Client Acc</h3>
          <p className="text-3xl font-bold text-green-900">
            {data.summary.avg_final_client_acc?.toFixed(2)}%
          </p>
        </div>
        
        <div className="bg-purple-50 p-6 rounded-lg border border-purple-200">
          <h3 className="text-sm font-medium text-purple-600 mb-2">Avg Final Global Acc</h3>
          <p className="text-3xl font-bold text-purple-900">
            {data.summary.avg_final_global_acc?.toFixed(2)}%
          </p>
        </div>
      </div>

      {data.summary.best_experiment && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-emerald-50 p-6 rounded-lg border border-emerald-200">
            <h3 className="text-sm font-medium text-emerald-600 mb-2">Best Experiment</h3>
            <p className="text-lg font-semibold text-emerald-900">{data.summary.best_experiment.name}</p>
            <p className="text-2xl font-bold text-emerald-900 mt-2">
              {data.summary.best_experiment.final_acc?.toFixed(2)}%
            </p>
          </div>
          
          {data.summary.avg_baseline_acc && (
            <div className="bg-amber-50 p-6 rounded-lg border border-amber-200">
              <h3 className="text-sm font-medium text-amber-600 mb-2">Avg Improvement over Baseline</h3>
              <p className="text-2xl font-bold text-amber-900">
                +{data.summary.avg_improvement_over_baseline?.toFixed(2)}%
              </p>
              <p className="text-sm text-amber-700 mt-1">
                Baseline: {data.summary.avg_baseline_acc?.toFixed(2)}%
              </p>
            </div>
          )}
        </div>
      )}

      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Final Accuracy Comparison</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={finalComparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="Client Acc" fill="#2563eb" />
            <Bar dataKey="Global Acc" fill="#7c3aed" />
            {data.experiments.some(e => e.baseline_acc) && (
              <Bar dataKey="Baseline" fill="#94a3b8" />
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Experiments List</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rounds</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Clients</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Final Client Acc</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Final Global Acc</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Baseline</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Improvement</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.experiments.map((exp, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">{exp.exp_name}</td>
                  <td className="px-4 py-3 text-sm text-gray-900">{exp.num_rounds}</td>
                  <td className="px-4 py-3 text-sm text-gray-900">{exp.num_clients}</td>
                  <td className="px-4 py-3 text-sm text-gray-900">{exp.final_avg_client_acc.toFixed(2)}%</td>
                  <td className="px-4 py-3 text-sm text-gray-900">{exp.final_global_test_acc.toFixed(2)}%</td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {exp.baseline_acc ? `${exp.baseline_acc.toFixed(2)}%` : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {exp.improvement_over_baseline ? (
                      <span className={exp.improvement_over_baseline > 0 ? 'text-green-600' : 'text-red-600'}>
                        {exp.improvement_over_baseline > 0 ? '+' : ''}{exp.improvement_over_baseline.toFixed(2)}%
                      </span>
                    ) : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <button
                      onClick={() => {
                        setSelectedExpDetail(exp.exp_name);
                        setViewMode('details');
                      }}
                      className="text-blue-600 hover:text-blue-800 font-medium"
                    >
                      Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  // Comparison section
  const renderComparison = () => (
    <div className="space-y-6">
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <h3 className="text-sm font-semibold mb-3">Select Experiments to Compare</h3>
        <div className="flex flex-wrap gap-2">
          {data.experiments.map((exp, idx) => (
            <button
              key={idx}
              onClick={() => toggleExperiment(exp.exp_name)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                selectedExperiments.includes(exp.exp_name)
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {exp.exp_name}
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Selected: {selectedExperiments.length} / {data.experiments.length}
        </p>
      </div>

      {selectedExps.length > 0 && (
        <>
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-4">Average Client Accuracy Over Rounds</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={accuracyComparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                {selectedExps.map((exp, idx) => (
                  <Line
                    key={exp.exp_name}
                    type="monotone"
                    dataKey={exp.exp_name}
                    stroke={COLORS[idx % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-4">Global Test Accuracy Over Rounds</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={globalAccuracyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                {selectedExps.map((exp, idx) => (
                  <Line
                    key={exp.exp_name}
                    type="monotone"
                    dataKey={exp.exp_name}
                    stroke={COLORS[idx % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-4">Average Labeled Samples Over Rounds</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={labeledSamplesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Labeled Samples', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                {selectedExps.map((exp, idx) => (
                  <Line
                    key={exp.exp_name}
                    type="monotone"
                    dataKey={exp.exp_name}
                    stroke={COLORS[idx % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );

  // Details section for a single experiment
  const renderDetails = () => {
    const exp = data.experiments.find(e => e.exp_name === selectedExpDetail);
    if (!exp) return <div>Experiment not found</div>;

    const veracityUsedData = exp.avg_veracity_used.map((val, idx) => ({
      round: idx,
      used: val,
      filtered: exp.avg_veracity_filtered[idx] || 0
    }));

    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold">{exp.exp_name}</h2>
          <button
            onClick={() => setViewMode('overview')}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Back to Overview
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <h3 className="text-xs font-medium text-blue-600 mb-1">Rounds</h3>
            <p className="text-2xl font-bold text-blue-900">{exp.num_rounds}</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <h3 className="text-xs font-medium text-purple-600 mb-1">Clients</h3>
            <p className="text-2xl font-bold text-purple-900">{exp.num_clients}</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <h3 className="text-xs font-medium text-green-600 mb-1">Best Client Acc</h3>
            <p className="text-2xl font-bold text-green-900">{exp.best_avg_client_acc.toFixed(2)}%</p>
          </div>
          <div className="bg-emerald-50 p-4 rounded-lg border border-emerald-200">
            <h3 className="text-xs font-medium text-emerald-600 mb-1">Best Global Acc</h3>
            <p className="text-2xl font-bold text-emerald-900">{exp.best_global_test_acc.toFixed(2)}%</p>
          </div>
        </div>

        {exp.baseline_acc && (
          <div className="bg-amber-50 p-6 rounded-lg border border-amber-200">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-sm font-medium text-amber-600 mb-1">Baseline Comparison</h3>
                <p className="text-lg text-amber-900">
                  Baseline: {exp.baseline_acc.toFixed(2)}% → Final: {exp.final_avg_client_acc.toFixed(2)}%
                </p>
              </div>
              <div className="text-right">
                <p className="text-3xl font-bold text-amber-900">
                  {exp.improvement_over_baseline && exp.improvement_over_baseline > 0 ? '+' : ''}
                  {exp.improvement_over_baseline?.toFixed(2)}%
                </p>
                <p className="text-xs text-amber-600">Improvement</p>
              </div>
            </div>
          </div>
        )}

        {veracityUsedData.some(d => d.used > 0 || d.filtered > 0) && (
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-4">Veracity Feedback Usage</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={veracityUsedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="used" fill="#10b981" name="Veracity Used" />
                <Bar dataKey="filtered" fill="#ef4444" name="Veracity Filtered" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {exp.train_class_distribution && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Train Class Distribution</h3>
              <div className="space-y-2">
                {Object.entries(exp.train_class_distribution).map(([cls, count]) => (
                  <div key={cls} className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Class {cls}</span>
                    <span className="font-medium">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {exp.test_class_distribution && (
              <div className="bg-white p-6 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-4">Test Class Distribution</h3>
                <div className="space-y-2">
                  {Object.entries(exp.test_class_distribution).map(([cls, count]) => (
                    <div key={cls} className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Class {cls}</span>
                      <span className="font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-2">Experiment Path</h3>
          <p className="text-sm text-gray-600 font-mono bg-gray-50 p-3 rounded">{exp.exp_path}</p>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Federated Learning Dashboard</h1>
          <p className="text-gray-600 mb-4">
            Analyze and compare federated learning experiment results
          </p>
          
          {data.summary.total_experiments === 0 ? (
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Load Experiment Data</h3>
              <p className="text-sm text-gray-500 mb-4">
                Upload the JSON file generated by analyze_federated_results.py
              </p>
              <label className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer">
                <input
                  type="file"
                  accept=".json"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                Choose File
              </label>
            </div>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('overview')}
                className={`px-4 py-2 rounded font-medium ${
                  viewMode === 'overview'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Overview
              </button>
              <button
                onClick={() => setViewMode('comparison')}
                className={`px-4 py-2 rounded font-medium ${
                  viewMode === 'comparison'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Comparison
              </button>
              <label className="ml-auto inline-flex items-center px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 cursor-pointer">
                <input
                  type="file"
                  accept=".json"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                Load New Data
              </label>
            </div>
          )}
        </div>

        {data.summary.total_experiments > 0 && (
          <div>
            {viewMode === 'overview' && renderOverview()}
            {viewMode === 'comparison' && renderComparison()}
            {viewMode === 'details' && renderDetails()}
          </div>
        )}
      </div>
    </div>
  );
}
