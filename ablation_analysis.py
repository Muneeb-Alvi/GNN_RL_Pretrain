#!/usr/bin/env python3
"""
Ablation Study Analysis Script
==============================

This script provides comprehensive analysis tools for comparing GNN model performance
in the ablation study. It loads trained models, runs inference on test data, and 
generates detailed performance comparisons.

Usage:
    python ablation_analysis.py --models_dir checkpoints/ --data_dir gnn_pickles/val/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Add project root to path
ROOT_DIR = "/scratch/maa9509/GNN_RL_Pretrain"
sys.path.insert(0, ROOT_DIR)

from utils_gnn.data_utils import GNNDatasetParallel
from utils_gnn.modeling import SimpleGCN, SimpleGAT, PearlGATv2, ResidualGIN, SimpleGraphSAGE, DeepGraphSAGE
from utils_gnn.train_utils import safe_mape_metric, smape_metric
from torch_geometric.loader import DataLoader


class AblationAnalyzer:
    def __init__(self, models_dir, data_dir, device='cuda:0'):
        """
        Initialize the ablation study analyzer
        
        Args:
            models_dir: Directory containing trained model checkpoints
            data_dir: Directory containing processed validation data
            device: Device to run inference on
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        self.models = {}
        self.results = {}
        
        # Load dataset
        print("Loading validation dataset...")
        self.dataset = GNNDatasetParallel(
            pkl_output_folder=str(data_dir),
            just_load_pickled=True
        )
        print(f"Loaded {len(self.dataset)} samples")
        
        # Get all attributes for analysis
        self.attrs = self.dataset.get_all_attrs()
        print(f"Loaded {len(self.attrs)} attribute records")
        
        # Create dataloader
        self.dataloader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        
    def load_model(self, model_name, checkpoint_path, input_size=17, hidden_size=256):
        """Load a trained model from checkpoint"""
        print(f"Loading {model_name} from {checkpoint_path}")
        
        # Create model architecture
        if model_name == "SimpleGraphSAGE":
            model = SimpleGraphSAGE(
                in_channels=input_size,
                hidden_channels=hidden_size,
                num_layers=4,
                dropout=0.1,
                out_channels=1
            )
        elif model_name == "DeepGraphSAGE":
            model = DeepGraphSAGE(
                in_channels=input_size,
                hidden_channels=hidden_size,
                num_layers=6,
                dropout=0.1,
                out_channels=1
            )
        elif model_name == "SimpleGCN":
            model = SimpleGCN(
                in_channels=input_size,
                hidden_channels=hidden_size,
                out_channels=1
            )
        elif model_name == "SimpleGAT":
            model = SimpleGAT(
                in_channels=input_size,
                hidden_channels=hidden_size,
                out_channels=1
            )
        elif model_name == "PearlGATv2":
            model = PearlGATv2(
                in_channels=input_size,
                hidden_channels=hidden_size,
                num_heads=4,
                dropout=0.1,
                out_channels=1
            )
        elif model_name == "ResidualGIN":
            model = ResidualGIN(
                in_channels=input_size,
                hidden_channels=hidden_size,
                out_channels=1
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        return model
    
    def run_inference(self, model_name):
        """Run inference with a model and collect predictions"""
        print(f"Running inference with {model_name}...")
        
        model = self.models[model_name]
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                batch = batch.to(self.device)
                outputs = model(batch)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        pred_tensor = torch.tensor(predictions)
        target_tensor = torch.tensor(targets)
        
        mse_loss = torch.nn.functional.mse_loss(pred_tensor, target_tensor).item()
        mape_safe = safe_mape_metric(pred_tensor, target_tensor).mean().item()
        smape = smape_metric(pred_tensor, target_tensor).mean().item()
        
        self.results[model_name] = {
            'predictions': predictions,
            'targets': targets,
            'mse_loss': mse_loss,
            'mape_safe': mape_safe,
            'smape': smape
        }
        
        print(f"{model_name} - MSE: {mse_loss:.4f}, MAPE_safe: {mape_safe:.2f}%, SMAPE: {smape:.2f}%")
        return predictions, targets
    
    def create_results_dataframe(self):
        """Create a comprehensive DataFrame with predictions and attributes"""
        if not self.results:
            raise ValueError("No results available. Run inference first.")
        
        # Start with attributes
        attr_data = []
        for i, attr in enumerate(self.attrs):
            func_name, sched_id, sched_str, exec_time, memory_use, node_name, tree_footprint, actual_speedup = attr
            attr_data.append({
                'sample_idx': i,
                'func_name': func_name,
                'sched_id': sched_id,
                'sched_str': sched_str,
                'exec_time': exec_time,
                'memory_use': memory_use,
                'node_name': node_name,
                'tree_footprint': tree_footprint,
                'actual_speedup': actual_speedup,
                'target': self.results[list(self.results.keys())[0]]['targets'][i]
            })
        
        df = pd.DataFrame(attr_data)
        
        # Add predictions from each model
        for model_name, results in self.results.items():
            df[f'{model_name}_pred'] = results['predictions']
            df[f'{model_name}_error'] = np.abs(results['predictions'] - results['targets'])
            df[f'{model_name}_rel_error'] = np.abs(results['predictions'] - results['targets']) / np.maximum(results['targets'], 1.0)
        
        return df
    
    def generate_summary_table(self):
        """Generate summary performance table"""
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'MSE Loss': results['mse_loss'],
                'MAPE_safe (%)': results['mape_safe'],
                'SMAPE (%)': results['smape'],
                'Samples': len(results['predictions'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('MAPE_safe (%)')
        return summary_df
    
    def analyze_by_program_characteristics(self, df):
        """Analyze performance by program characteristics"""
        analyses = {}
        
        # Performance by function name (top 20 most common)
        func_counts = df['func_name'].value_counts().head(20)
        func_analysis = []
        
        for func_name in func_counts.index:
            func_data = df[df['func_name'] == func_name]
            func_stats = {'func_name': func_name, 'count': len(func_data)}
            
            for model_name in self.results.keys():
                mape_col = f'{model_name}_rel_error'
                func_stats[f'{model_name}_mean_rel_error'] = func_data[mape_col].mean()
                
            func_analysis.append(func_stats)
        
        analyses['by_function'] = pd.DataFrame(func_analysis)
        
        # Performance by memory usage quartiles
        df['memory_quartile'] = pd.qcut(df['memory_use'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        memory_analysis = []
        
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            quartile_data = df[df['memory_quartile'] == quartile]
            quartile_stats = {'memory_quartile': quartile, 'count': len(quartile_data)}
            
            for model_name in self.results.keys():
                mape_col = f'{model_name}_rel_error'
                quartile_stats[f'{model_name}_mean_rel_error'] = quartile_data[mape_col].mean()
                
            memory_analysis.append(quartile_stats)
        
        analyses['by_memory'] = pd.DataFrame(memory_analysis)
        
        # Performance by speedup ranges
        df['speedup_range'] = pd.cut(df['actual_speedup'], 
                                   bins=[0, 1.5, 3, 10, float('inf')], 
                                   labels=['Low (≤1.5x)', 'Medium (1.5-3x)', 'High (3-10x)', 'Very High (>10x)'])
        
        speedup_analysis = []
        for speedup_range in df['speedup_range'].cat.categories:
            range_data = df[df['speedup_range'] == speedup_range]
            if len(range_data) == 0:
                continue
                
            range_stats = {'speedup_range': speedup_range, 'count': len(range_data)}
            
            for model_name in self.results.keys():
                mape_col = f'{model_name}_rel_error'
                range_stats[f'{model_name}_mean_rel_error'] = range_data[mape_col].mean()
                
            speedup_analysis.append(range_stats)
        
        analyses['by_speedup'] = pd.DataFrame(speedup_analysis)
        
        return analyses
    
    def create_visualizations(self, df, output_dir='ablation_analysis'):
        """Create visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model performance comparison
        summary_df = self.generate_summary_table()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MSE Loss comparison
        axes[0].bar(summary_df['Model'], summary_df['MSE Loss'])
        axes[0].set_title('MSE Loss Comparison')
        axes[0].set_ylabel('MSE Loss')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAPE_safe comparison
        axes[1].bar(summary_df['Model'], summary_df['MAPE_safe (%)'])
        axes[1].set_title('MAPE_safe Comparison')
        axes[1].set_ylabel('MAPE_safe (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # SMAPE comparison
        axes[2].bar(summary_df['Model'], summary_df['SMAPE (%)'])
        axes[2].set_title('SMAPE Comparison')
        axes[2].set_ylabel('SMAPE (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction vs Target scatter plots
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            predictions = results['predictions']
            targets = results['targets']
            
            # Scatter plot
            ax.scatter(targets, predictions, alpha=0.5, s=1)
            
            # Perfect prediction line
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('True Speedup')
            ax.set_ylabel('Predicted Speedup')
            ax.set_title(f'{model_name}\nMAPE_safe: {results["mape_safe"]:.2f}%')
            
            # Add correlation coefficient
            correlation = np.corrcoef(targets, predictions)[0, 1]
            ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def save_results(self, output_dir='ablation_analysis'):
        """Save all analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save summary table
        summary_df = self.generate_summary_table()
        summary_df.to_csv(output_dir / 'model_summary.csv', index=False)
        
        # Save detailed results DataFrame
        df = self.create_results_dataframe()
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        
        # Save characteristic analyses
        analyses = self.analyze_by_program_characteristics(df)
        for analysis_name, analysis_df in analyses.items():
            analysis_df.to_csv(output_dir / f'analysis_{analysis_name}.csv', index=False)
        
        # Save raw results as JSON
        results_json = {}
        for model_name, results in self.results.items():
            results_json[model_name] = {
                'mse_loss': float(results['mse_loss']),
                'mape_safe': float(results['mape_safe']),
                'smape': float(results['smape']),
                'predictions': results['predictions'].tolist(),
                'targets': results['targets'].tolist()
            }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
        
        return df, analyses


def main():
    parser = argparse.ArgumentParser(description='GNN Ablation Study Analysis')
    parser.add_argument('--models_dir', default='checkpoints/', 
                       help='Directory containing model checkpoints')
    parser.add_argument('--data_dir', default='gnn_pickles/val/', 
                       help='Directory containing validation data')
    parser.add_argument('--output_dir', default='ablation_analysis/', 
                       help='Output directory for results')
    parser.add_argument('--device', default='cuda:0', 
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AblationAnalyzer(args.models_dir, args.data_dir, args.device)
    
    # Define model checkpoints (update these paths based on your actual checkpoints)
    model_checkpoints = {
        'SimpleGraphSAGE': 'SimpleGraphSAGE_1M_dataset_best.pt',
        'DeepGraphSAGE': 'DeepGraphSAGE_1M_dataset_best.pt', 
        'SimpleGCN': 'SimpleGCN_1M_dataset_best.pt',
        'SimpleGAT': 'SimpleGAT_1M_dataset_best.pt',
        'ResidualGIN': 'ResidualGIN_1M_dataset_best.pt',
        'PearlGATv2': 'PearlGATv2_1M_dataset_best.pt'
    }
    
    # Load models and run inference
    for model_name, checkpoint_name in model_checkpoints.items():
        checkpoint_path = Path(args.models_dir) / checkpoint_name
        if checkpoint_path.exists():
            try:
                analyzer.load_model(model_name, checkpoint_path)
                analyzer.run_inference(model_name)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    if not analyzer.results:
        print("No models successfully loaded and evaluated!")
        return
    
    # Generate analysis
    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS")
    print("="*50)
    
    # Print summary table
    summary_df = analyzer.generate_summary_table()
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    
    # Save results and create visualizations
    df, analyses = analyzer.save_results(args.output_dir)
    analyzer.create_visualizations(df, args.output_dir)
    
    # Print top insights
    print(f"\n📊 Best performing model (MAPE_safe): {summary_df.iloc[0]['Model']} ({summary_df.iloc[0]['MAPE_safe (%)']:.2f}%)")
    print(f"📊 Total samples analyzed: {len(df)}")
    print(f"📊 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()







