#!/usr/bin/env python3
"""
Dragon Tower Data Analysis
Analyzes game logs to determine patterns and prediction feasibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_data(filepath='dragon_tower_logs/combined.csv'):
    """Load and parse the combined game data"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} total tile records")
    print(f"Number of unique games: {df['GAME'].nunique()}")
    print(f"Rows per game: {df.groupby('GAME')['ROW'].nunique().iloc[0]}")
    return df

def calculate_distributions(df):
    """Calculate tile distribution statistics"""
    print("\n" + "="*60)
    print("TILE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Overall distribution
    total_tiles = len(df)
    left_correct = df['LeftTile'].sum()
    right_correct = df['RightTile'].sum()
    
    print(f"\nOverall Statistics:")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Left tile correct: {left_correct} ({left_correct/total_tiles*100:.2f}%)")
    print(f"  Right tile correct: {right_correct} ({right_correct/total_tiles*100:.2f}%)")
    
    # Per-row distribution
    print(f"\nPer-Row Distribution:")
    row_stats = df.groupby('ROW').agg({
        'LeftTile': ['sum', 'mean'],
        'RightTile': ['sum', 'mean']
    })
    
    for row in sorted(df['ROW'].unique()):
        row_data = df[df['ROW'] == row]
        left_prob = row_data['LeftTile'].mean()
        right_prob = row_data['RightTile'].mean()
        n_samples = len(row_data)
        print(f"  Row {row}: Left={left_prob*100:.2f}% Right={right_prob*100:.2f}% (n={n_samples})")
    
    return row_stats

def test_randomness(df):
    """Perform statistical tests for randomness"""
    print("\n" + "="*60)
    print("RANDOMNESS TESTS")
    print("="*60)
    
    # Chi-square test for each row
    print("\nChi-Square Tests (H0: tiles are 50/50 random):")
    for row in sorted(df['ROW'].unique()):
        row_data = df[df['ROW'] == row]
        left_count = row_data['LeftTile'].sum()
        right_count = row_data['RightTile'].sum()
        
        # Chi-square test
        observed = [left_count, right_count]
        expected_freq = [len(row_data)/2, len(row_data)/2]
        chi2, p_value = stats.chisquare(observed, expected_freq)
        
        significant = "✗ BIAS DETECTED" if p_value < 0.05 else "✓ Random"
        print(f"  Row {row}: χ²={chi2:.2f}, p={p_value:.4f} {significant}")
    
    # Overall test
    left_total = df['LeftTile'].sum()
    right_total = df['RightTile'].sum()
    chi2, p_value = stats.chisquare([left_total, right_total], [len(df)/2, len(df)/2])
    print(f"\n  Overall: χ²={chi2:.2f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("  ⚠️  Data shows significant deviation from 50/50 distribution!")
    else:
        print("  ✓ Data is consistent with random 50/50 distribution")

def analyze_patterns(df):
    """Look for sequential patterns and dependencies"""
    print("\n" + "="*60)
    print("PATTERN ANALYSIS")
    print("="*60)
    
    # Prepare data by game
    games = []
    for game_id in df['GAME'].unique():
        game_data = df[df['GAME'] == game_id].sort_values('ROW')
        tiles = game_data['LeftTile'].values  # 1 = left, 0 = right
        games.append(tiles)
    
    games = np.array(games)
    
    # Check for row-to-row correlations
    print("\nRow-to-Row Correlations:")
    print("(Checking if tile choice in one row predicts the next row)")
    
    correlations = []
    for row in range(1, 9):  # Rows 1-8 (predicting row 2-9)
        current_row = games[:, row-1]
        next_row = games[:, row]
        corr, p_value = stats.pearsonr(current_row, next_row)
        correlations.append(corr)
        
        sig = "**" if abs(p_value) < 0.05 else ""
        print(f"  Row {row} → Row {row+1}: r={corr:.4f}, p={p_value:.4f} {sig}")
    
    avg_corr = np.mean(np.abs(correlations))
    print(f"\n  Average absolute correlation: {avg_corr:.4f}")
    
    if avg_corr > 0.1:
        print("  ⚠️  Moderate correlations detected - patterns may exist!")
    else:
        print("  ✓ Weak correlations - tiles appear independent")
    
    # Check for streaks
    print("\nStreak Analysis:")
    all_streaks = []
    for game in games:
        streak = 1
        for i in range(1, len(game)):
            if game[i] == game[i-1]:
                streak += 1
            else:
                all_streaks.append(streak)
                streak = 1
        all_streaks.append(streak)
    
    print(f"  Average streak length: {np.mean(all_streaks):.2f}")
    print(f"  Max streak observed: {np.max(all_streaks)}")
    print(f"  Expected for random: ~2.0")
    
    return correlations, games

def analyze_conditional_probabilities(df):
    """Analyze conditional probabilities based on previous tiles"""
    print("\n" + "="*60)
    print("CONDITIONAL PROBABILITY ANALYSIS")
    print("="*60)
    
    # Prepare game data
    games = []
    for game_id in df['GAME'].unique():
        game_data = df[df['GAME'] == game_id].sort_values('ROW')
        tiles = game_data['LeftTile'].values
        games.append(tiles)
    
    games = np.array(games)
    
    # For each row, check if the result depends on previous row(s)
    print("\nP(Left | Previous Row):")
    for row in range(2, 10):  # Rows 2-9
        prev_row = games[:, row-2]
        curr_row = games[:, row-1]
        
        # P(curr=Left | prev=Left)
        prob_L_given_L = curr_row[prev_row == 1].mean()
        # P(curr=Left | prev=Right)
        prob_L_given_R = curr_row[prev_row == 0].mean()
        
        diff = abs(prob_L_given_L - prob_L_given_R)
        
        print(f"  Row {row}:")
        print(f"    P(Left|Prev=Left)  = {prob_L_given_L*100:.1f}%")
        print(f"    P(Left|Prev=Right) = {prob_L_given_R*100:.1f}%")
        print(f"    Difference: {diff*100:.1f}%")
        
        if diff > 0.1:
            print(f"    ⚠️  Significant dependency detected!")

def create_visualizations(df, correlations, games):
    """Create visualization plots"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Tile distribution per row
    row_probs = df.groupby('ROW')['LeftTile'].mean()
    axes[0, 0].bar(row_probs.index, row_probs.values, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', label='50% (Random)')
    axes[0, 0].set_xlabel('Row')
    axes[0, 0].set_ylabel('Probability of Left Tile')
    axes[0, 0].set_title('Tile Distribution by Row')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Heatmap of tile positions
    heatmap_data = np.zeros((9, 2))
    for row in range(1, 10):
        row_data = df[df['ROW'] == row]
        heatmap_data[row-1, 0] = row_data['LeftTile'].mean()
        heatmap_data[row-1, 1] = row_data['RightTile'].mean()
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=['Left', 'Right'], yticklabels=range(1, 10),
                ax=axes[0, 1], vmin=0, vmax=1)
    axes[0, 1].set_title('Probability Heatmap')
    axes[0, 1].set_ylabel('Row')
    
    # 3. Correlation between consecutive rows
    if correlations:
        axes[1, 0].plot(range(1, 9), correlations, marker='o', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].set_xlabel('Row N')
        axes[1, 0].set_ylabel('Correlation with Row N+1')
        axes[1, 0].set_title('Sequential Row Correlations')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sample game patterns
    sample_games = games[:20]  # First 20 games
    axes[1, 1].imshow(sample_games, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    axes[1, 1].set_xlabel('Row')
    axes[1, 1].set_ylabel('Game Number')
    axes[1, 1].set_title('Sample Game Patterns (Green=Left, Red=Right)')
    axes[1, 1].set_xticks(range(9))
    axes[1, 1].set_xticklabels(range(1, 10))
    
    plt.tight_layout()
    output_file = 'dragon_tower_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")
    
def generate_report(df):
    """Generate final prediction feasibility report"""
    print("\n" + "="*60)
    print("PREDICTION FEASIBILITY REPORT")
    print("="*60)
    
    # Calculate baseline accuracy
    baseline = max(df['LeftTile'].mean(), df['RightTile'].mean())
    
    print(f"\nBaseline Prediction Accuracy:")
    print(f"  Always predict most common: {baseline*100:.2f}%")
    print(f"  Random guessing: ~50%")
    
    # Check for any row with strong bias
    strong_bias = False
    for row in range(1, 10):
        row_data = df[df['ROW'] == row]
        left_prob = row_data['LeftTile'].mean()
        if abs(left_prob - 0.5) > 0.15:  # More than 15% deviation
            strong_bias = True
            print(f"  ⚠️  Row {row} has strong bias: {max(left_prob, 1-left_prob)*100:.2f}%")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    # Overall assessment
    left_overall = df['LeftTile'].mean()
    if abs(left_overall - 0.5) < 0.05 and not strong_bias:
        print("\n✓ The data appears to be truly random (50/50 distribution).")
        print("✓ No significant patterns or dependencies detected.")
        print("\n⚠️  PREDICTION FEASIBILITY: LOW")
        print("   ML models will likely perform at ~50% accuracy (random baseline).")
        print("   However, we can still build models to:")
        print("   - Detect any subtle biases per row")
        print("   - Provide probability-based recommendations")
        print("   - Track statistics during gameplay")
    else:
        print("\n⚠️  The data shows some non-random characteristics!")
        print("✓ PREDICTION FEASIBILITY: MODERATE")
        print("   ML models may achieve >50% accuracy by learning:")
        print("   - Positional biases")
        print("   - Sequential patterns")
        print("   - Conditional dependencies")

def main():
    """Main analysis pipeline"""
    print("="*60)
    print("DRAGON TOWER DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Run analyses
    row_stats = calculate_distributions(df)
    test_randomness(df)
    correlations, games = analyze_patterns(df)
    analyze_conditional_probabilities(df)
    create_visualizations(df, correlations, games)
    generate_report(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print("  1. Review 'dragon_tower_analysis.png' for visual insights")
    print("  2. Run 'python dragon_tower_predictor.py' to train ML models")
    print("  3. Use 'python predict_game.py' for live predictions")

if __name__ == "__main__":
    main()
