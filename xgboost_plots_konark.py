import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class NBAModelVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_all_metrics(self, y_true, y_pred_proba, feature_importance_df=None, save_prefix='nba_model'):
        """Generate all visualization plots"""
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. ROC Curve
        ax1 = plt.subplot(4, 2, 1)
        self.plot_roc_curve(y_true, y_pred_proba, ax=ax1)
        
        # 2. Calibration Plot
        ax2 = plt.subplot(4, 2, 2)
        self.plot_calibration_curve(y_true, y_pred_proba, ax=ax2)
        
        # 3. Probability Distribution
        ax3 = plt.subplot(4, 2, 3)
        self.plot_probability_distribution(y_true, y_pred_proba, ax=ax3)
        
        # 4. Confusion Matrix
        ax4 = plt.subplot(4, 2, 4)
        self.plot_confusion_matrix(y_true, y_pred_proba, ax=ax4)
        
        # 5. Feature Importance
        if feature_importance_df is not None:
            ax5 = plt.subplot(4, 2, 5)
            self.plot_feature_importance(feature_importance_df, ax=ax5)
        
        # 6. Profit/Loss Simulation
        ax6 = plt.subplot(4, 2, 6)
        self.plot_betting_simulation(y_true, y_pred_proba, ax=ax6)
        
        # 7. Probability Threshold Analysis
        ax7 = plt.subplot(4, 2, 7)
        self.plot_threshold_analysis(y_true, y_pred_proba, ax=ax7)
        
        # 8. Expected Value by Probability Range
        ax8 = plt.subplot(4, 2, 8)
        self.plot_ev_by_probability(y_true, y_pred_proba, ax=ax8)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curve(self, y_true, y_pred_proba, ax=None):
        """Plot ROC curve"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Model Performance')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
    def plot_calibration_curve(self, y_true, y_pred_proba, n_bins=10, ax=None):
        """Plot calibration curve"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Model calibration
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label='Model calibration', color='red', markersize=8)
        
        # Add calibration bars
        for i in range(len(mean_predicted_value)):
            ax.plot([mean_predicted_value[i], mean_predicted_value[i]], 
                   [mean_predicted_value[i], fraction_of_positives[i]], 
                   'r-', alpha=0.3, linewidth=2)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives (Actual Win Rate)')
        ax.set_title('Calibration Plot - Probability Reliability')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
    def plot_probability_distribution(self, y_true, y_pred_proba, ax=None):
        """Plot distribution of predicted probabilities"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        # Separate predictions by actual outcome
        home_wins = y_pred_proba[y_true == 1]
        home_losses = y_pred_proba[y_true == 0]
        
        # Plot histograms
        ax.hist(home_losses, bins=30, alpha=0.5, label='Actual Away Wins', color='blue', density=True)
        ax.hist(home_wins, bins=30, alpha=0.5, label='Actual Home Wins', color='red', density=True)
        
        ax.set_xlabel('Predicted Home Win Probability')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Predictions by Actual Outcome')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at 0.5
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
        
    def plot_confusion_matrix(self, y_true, y_pred_proba, threshold=0.5, ax=None):
        """Plot confusion matrix"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Away Win', 'Home Win'],
                    yticklabels=['Away Win', 'Home Win'])
        
        # Add percentages
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.1%})',
                       ha='center', va='center', fontsize=10, color='darkblue')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (threshold={threshold})')
        
    def plot_feature_importance(self, feature_importance_df, top_n=15, ax=None):
        """Plot feature importance"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        # Get top features
        top_features = feature_importance_df.head(top_n)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='skyblue', edgecolor='navy')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')
            
    def plot_betting_simulation(self, y_true, y_pred_proba, stake=100, ax=None):
        """Simulate betting returns at different probability thresholds"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        thresholds = np.arange(0.5, 0.8, 0.01)
        returns = []
        bet_counts = []
        
        for threshold in thresholds:
            # Only bet when probability exceeds threshold
            bet_mask = y_pred_proba >= threshold
            
            if bet_mask.sum() == 0:
                returns.append(0)
                bet_counts.append(0)
                continue
                
            # Calculate returns (assuming -110 odds for simplicity)
            wins = (y_true[bet_mask] == 1).sum()
            losses = (~(y_true[bet_mask] == 1)).sum()
            
            # Standard -110 betting odds
            profit = wins * (stake * 100/110) - losses * stake
            returns.append(profit)
            bet_counts.append(bet_mask.sum())
        
        # Plot returns
        ax2 = ax.twinx()
        
        ax.plot(thresholds, returns, 'b-', linewidth=2, label='Profit/Loss')
        ax2.plot(thresholds, bet_counts, 'r--', linewidth=2, label='Number of Bets')
        
        ax.set_xlabel('Minimum Probability Threshold')
        ax.set_ylabel('Profit/Loss ($)', color='b')
        ax2.set_ylabel('Number of Bets', color='r')
        ax.set_title('Betting Simulation - Returns by Confidence Threshold')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.grid(True, alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
    def plot_threshold_analysis(self, y_true, y_pred_proba, ax=None):
        """Analyze metrics at different probability thresholds"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        thresholds = np.arange(0.3, 0.8, 0.01)
        accuracies = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = (y_pred == y_true).mean()
            
            # Precision: Of all positive predictions, how many were correct?
            if (y_pred == 1).sum() > 0:
                precision = ((y_pred == 1) & (y_true == 1)).sum() / (y_pred == 1).sum()
            else:
                precision = 0
                
            # Recall: Of all actual positives, how many did we predict?
            if (y_true == 1).sum() > 0:
                recall = ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
            else:
                recall = 0
                
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
        
        # Plot
        ax.plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
        ax.plot(thresholds, precisions, 'r-', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
        
        ax.set_xlabel('Probability Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Metrics by Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.3, 0.8])
        ax.set_ylim([0, 1])
        
        # Add optimal threshold marker
        optimal_idx = np.argmax(accuracies)
        ax.plot(thresholds[optimal_idx], accuracies[optimal_idx], 'ko', markersize=10)
        ax.annotate(f'Optimal: {thresholds[optimal_idx]:.2f}', 
                   xy=(thresholds[optimal_idx], accuracies[optimal_idx]),
                   xytext=(thresholds[optimal_idx] + 0.05, accuracies[optimal_idx] - 0.05),
                   arrowprops=dict(arrowstyle='->'))
        
    def plot_ev_by_probability(self, y_true, y_pred_proba, ax=None):
        """Plot expected value by probability range"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            
        # Create probability bins
        prob_bins = np.arange(0, 1.1, 0.1)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
        
        actual_win_rates = []
        counts = []
        expected_values = []
        
        for i in range(len(prob_bins) - 1):
            mask = (y_pred_proba >= prob_bins[i]) & (y_pred_proba < prob_bins[i+1])
            
            if mask.sum() > 0:
                actual_rate = y_true[mask].mean()
                actual_win_rates.append(actual_rate)
                counts.append(mask.sum())
                
                # Calculate EV assuming -110 odds
                ev = actual_rate * (100/110) - (1 - actual_rate)
                expected_values.append(ev)
            else:
                actual_win_rates.append(np.nan)
                counts.append(0)
                expected_values.append(np.nan)
        
        # Plot bars
        colors = ['red' if ev < 0 else 'green' for ev in expected_values]
        bars = ax.bar(bin_centers, expected_values, width=0.08, color=colors, alpha=0.7, edgecolor='black')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.01,
                       f'n={count}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Predicted Probability Range')
        ax.set_ylabel('Expected Value (per $1 bet)')
        ax.set_title('Expected Value by Probability Range (-110 odds)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim([-0.05, 1.05])
        
        # Add profitability threshold
        profitable_threshold = 110/210  # Break-even probability for -110 odds
        ax.axvline(x=profitable_threshold, color='orange', linestyle='--', 
                  label=f'Break-even: {profitable_threshold:.3f}')
        ax.legend()
        
    def plot_time_series_performance(self, test_df, y_true, y_pred_proba, ax=None):
        """Plot model performance over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
            
        # Add predictions to dataframe
        results_df = test_df.copy()
        results_df['predicted_prob'] = y_pred_proba
        results_df['correct'] = ((y_pred_proba >= 0.5) == y_true).astype(int)
        
        # Group by week
        results_df['week'] = pd.to_datetime(results_df['GAME_DATE']).dt.to_period('W')
        weekly_stats = results_df.groupby('week').agg({
            'correct': ['mean', 'count'],
            'predicted_prob': 'mean'
        })
        
        # Plot
        ax2 = ax.twinx()
        
        weeks = weekly_stats.index.astype(str)
        accuracy = weekly_stats['correct']['mean']
        counts = weekly_stats['correct']['count']
        
        ax.plot(weeks, accuracy, 'b-', linewidth=2, marker='o', label='Weekly Accuracy')
        ax2.bar(weeks, counts, alpha=0.3, color='gray', label='Games per Week')
        
        # Add rolling average
        rolling_acc = accuracy.rolling(window=4, min_periods=1).mean()
        ax.plot(weeks, rolling_acc, 'r--', linewidth=2, label='4-week Moving Avg')
        
        ax.set_xlabel('Week')
        ax.set_ylabel('Accuracy', color='b')
        ax2.set_ylabel('Number of Games', color='gray')
        ax.set_title('Model Performance Over Time')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax.grid(True, alpha=0.3)
        
        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_ylim([0.4, 0.7])

# Integration function to add to main predictor class
def visualize_model_performance(y_test, y_pred_proba, test_df, feature_importance_df):
    """Create comprehensive visualization of model performance"""
    
    visualizer = NBAModelVisualizer()
    
    # Generate all plots
    visualizer.plot_all_metrics(
        y_test, 
        y_pred_proba, 
        feature_importance_df,
        save_prefix='nba_model'
    )
    
    # Additional time series plot
    plt.figure(figsize=(14, 6))
    visualizer.plot_time_series_performance(test_df, y_test, y_pred_proba)
    plt.tight_layout()
    plt.savefig('nba_model_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return visualizer