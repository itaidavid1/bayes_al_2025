"""
Launch the federated learning dashboard to visualize experiment results.
Generates an HTML dashboard and opens it in your default browser.

Usage:
    python view_federated_dashboard.py <json_file> [--output dashboard.html]
"""

import argparse
import json
import os
import sys
import webbrowser
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate and launch federated learning dashboard"
    )
    parser.add_argument(
        "json_file",
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
    print("Federated Learning Dashboard Viewer")
    print("=" * 60)
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    # Validate JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n✓ Loaded: {json_path}")
        print(f"  Experiments: {data['summary']['total_experiments']}")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return 1
    
    # Generate HTML dashboard
    print(f"\nGenerating HTML dashboard...")
    script_dir = Path(__file__).parent
    generator_script = script_dir / "generate_html_dashboard.py"
    
    if not generator_script.exists():
        print(f"Error: Generator script not found: {generator_script}")
        return 1
    
    # Call the HTML generator
    try:
        subprocess.run([
            sys.executable,
            str(generator_script),
            str(json_path),
            "-o", args.output
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating dashboard: {e}")
        return 1
    
    output_path = Path(args.output).absolute()
    print(f"\n✓ Dashboard generated: {output_path}")
    
    # Open in browser
    print(f"\nOpening dashboard in browser...")
    try:
        webbrowser.open(f"file:///{output_path}")
        print("✓ Dashboard opened in browser")
    except Exception as e:
        print(f"Could not automatically open browser: {e}")
        print(f"\nPlease open manually: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
