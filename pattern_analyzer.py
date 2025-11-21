#!/usr/bin/env python3
"""
Dragon Tower Pattern Analyzer
Analyzes complete game patterns (512 possible combinations)
and provides pattern-based predictions
"""

import pandas as pd
import numpy as np
from collections import Counter
import json

class PatternAnalyzer:
    """Analyzes complete game patterns"""
    
    def __init__(self):
        self.pattern_frequencies = {}
        self.total_games = 0
        
    def load_and_analyze(self, filepath='dragon_tower_logs/combined.csv'):
        """Load data and analyze pattern frequencies"""
        print("="*60)
        print("PATTERN FREQUENCY ANALYSIS")
        print("="*60)
        print(f"\nTotal possible patterns: 512 (2^9)")
        
        df = pd.read_csv(filepath)
        
        # Extract complete patterns for each game
        patterns = []
        for game_id in df['GAME'].unique():
            game_data = df[df['GAME'] == game_id].sort_values('ROW')
            # Create pattern as tuple of tiles (1=left, 0=right)
            pattern = tuple(game_data['LeftTile'].values)
            patterns.append(pattern)
        
        self.total_games = len(patterns)
        self.pattern_frequencies = Counter(patterns)
        
        print(f"\nGames analyzed: {self.total_games}")
        print(f"Unique patterns observed: {len(self.pattern_frequencies)}")
        print(f"Pattern coverage: {len(self.pattern_frequencies)/512*100:.1f}% of all possible patterns")
        
        return patterns
    
    def display_top_patterns(self, n=20):
        """Display most common patterns"""
        print("\n" + "="*60)
        print(f"TOP {n} MOST COMMON PATTERNS")
        print("="*60)
        
        most_common = self.pattern_frequencies.most_common(n)
        
        print(f"\n{'Rank':<6} {'Pattern':<20} {'Count':<8} {'Probability':<12} {'Visual'}")
        print("-" * 70)
        
        for i, (pattern, count) in enumerate(most_common, 1):
            prob = count / self.total_games * 100
            visual = ''.join(['L' if t == 1 else 'R' for t in pattern])
            bar = '█' * int(prob / 2)  # Scale for display
            print(f"{i:<6} {visual:<20} {count:<8} {prob:>6.2f}%      {bar}")
        
        # Expected frequency for random distribution
        expected_freq = self.total_games / 512
        print(f"\nExpected frequency (if random): {expected_freq:.2f} games per pattern")
        print(f"Actual top pattern frequency: {most_common[0][1]} games")
        print(f"Ratio: {most_common[0][1] / expected_freq:.2f}x expected")
        
        if most_common[0][1] > expected_freq * 2:
            print("\n⚠️  Top pattern appears >2x more than expected!")
            print("   Strong evidence of pattern bias!")
        elif most_common[0][1] > expected_freq * 1.5:
            print("\n⚠️  Top pattern appears >1.5x more than expected")
            print("   Moderate pattern bias detected")
        else:
            print("\n✓ Pattern distribution appears relatively uniform")
    
    def get_pattern_probability(self, pattern):
        """Get probability of a specific pattern"""
        count = self.pattern_frequencies.get(pattern, 0)
        return count / self.total_games if self.total_games > 0 else 0
    
    def predict_next_tile_from_patterns(self, partial_pattern):
        """
        Predict next tile based on pattern analysis
        
        Args:
            partial_pattern: tuple of tiles observed so far
            
        Returns:
            prediction, probability, compatible_patterns_count
        """
        partial_len = len(partial_pattern)
        
        if partial_len >= 9:
            return None, None, 0  # Game complete
        
        # Find all patterns that match the partial pattern
        compatible_patterns = []
        for pattern, count in self.pattern_frequencies.items():
            if pattern[:partial_len] == partial_pattern:
                compatible_patterns.append((pattern, count))
        
        if not compatible_patterns:
            # No historical matches - default to 50/50
            return 1, 0.5, 0
        
        # Count how many compatible patterns have left vs right for next position
        next_pos = partial_len
        left_count = sum(count for pattern, count in compatible_patterns if pattern[next_pos] == 1)
        right_count = sum(count for pattern, count in compatible_patterns if pattern[next_pos] == 0)
        
        total = left_count + right_count
        prob_left = left_count / total if total > 0 else 0.5
        
        prediction = 1 if prob_left > 0.5 else 0
        confidence = max(prob_left, 1 - prob_left)
        
        return prediction, confidence, len(compatible_patterns)
    
    def predict_most_likely_complete_pattern(self):
        """Return the single most common pattern"""
        if not self.pattern_frequencies:
            return None, 0
        
        most_common = self.pattern_frequencies.most_common(1)[0]
        pattern, count = most_common
        probability = count / self.total_games
        
        return pattern, probability
    
    def analyze_pattern_statistics(self):
        """Analyze pattern distribution statistics"""
        print("\n" + "="*60)
        print("PATTERN DISTRIBUTION STATISTICS")
        print("="*60)
        
        frequencies = list(self.pattern_frequencies.values())
        
        print(f"\nPattern frequency statistics:")
        print(f"  Mean: {np.mean(frequencies):.2f} games/pattern")
        print(f"  Median: {np.median(frequencies):.2f} games/pattern")
        print(f"  Std Dev: {np.std(frequencies):.2f}")
        print(f"  Min: {np.min(frequencies)} games")
        print(f"  Max: {np.max(frequencies)} games")
        
        # Calculate entropy (measure of randomness)
        total = sum(frequencies)
        probs = [f/total for f in frequencies]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(512)  # Maximum for uniform distribution
        
        print(f"\n  Entropy: {entropy:.2f} bits")
        print(f"  Max entropy (uniform): {max_entropy:.2f} bits")
        print(f"  Entropy ratio: {entropy/max_entropy*100:.1f}%")
        
        if entropy / max_entropy > 0.95:
            print("\n  ✓ High entropy - patterns are well distributed")
        elif entropy / max_entropy > 0.85:
            print("\n  ⚠️  Moderate entropy - some pattern clustering")
        else:
            print("\n  ⚠️  Low entropy - significant pattern bias!")
    
    def save_pattern_data(self, filename='pattern_frequencies.json'):
        """Save pattern frequency data"""
        # Convert tuple keys to strings for JSON
        data = {
            'total_games': self.total_games,
            'patterns': {
                ''.join(['L' if t == 1 else 'R' for t in pattern]): count
                for pattern, count in self.pattern_frequencies.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Pattern data saved to: {filename}")
    
    def load_pattern_data(self, filename='pattern_frequencies.json'):
        """Load pattern frequency data"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.total_games = data['total_games']
        # Convert string keys back to tuples
        self.pattern_frequencies = {
            tuple([1 if c == 'L' else 0 for c in pattern_str]): count
            for pattern_str, count in data['patterns'].items()
        }

def compare_approaches():
    """Compare pattern-based vs tile-based prediction"""
    print("\n" + "="*60)
    print("PREDICTION APPROACH COMPARISON")
    print("="*60)
    
    print("\n📊 PATTERN-BASED PREDICTION:")
    print("  Pros:")
    print("    ✓ Leverages complete game patterns")
    print("    ✓ Can identify 'hot' patterns that repeat")
    print("    ✓ Narrows down possibilities as game progresses")
    print("    ✓ Useful if certain patterns are favored")
    print("  Cons:")
    print("    ✗ Requires exact pattern match")
    print("    ✗ May have sparse data for rare partial patterns")
    print("    ✗ Doesn't learn positional/sequential rules")
    
    print("\n🤖 TILE-BY-TILE (ML) PREDICTION:")
    print("  Pros:")
    print("    ✓ Learns positional tendencies")
    print("    ✓ Adapts to sequences and streaks")
    print("    ✓ Better generalization")
    print("    ✓ Works even with unseen partial patterns")
    print("  Cons:")
    print("    ✗ May miss exact pattern repetitions")
    print("    ✗ Treats game as sequential rather than holistic")
    
    print("\n💡 HYBRID APPROACH (RECOMMENDED):")
    print("    Use pattern-based when:")
    print("      • Partial pattern has strong historical match")
    print("      • High confidence from pattern analysis (>70%)")
    print("    Use ML-based when:")
    print("      • Partial pattern is rare/unseen")
    print("      • Pattern confidence is low")

def main():
    """Main analysis"""
    analyzer = PatternAnalyzer()
    
    # Load and analyze patterns
    patterns = analyzer.load_and_analyze()
    
    # Display top patterns
    analyzer.display_top_patterns(20)
    
    # Statistics
    analyzer.analyze_pattern_statistics()
    
    # Compare approaches
    compare_approaches()
    
    # Save pattern data
    analyzer.save_pattern_data()
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE: Pattern-Based Prediction")
    print("="*60)
    
    # Simulate a partial game
    example_partial = (1, 1, 0, 1)  # First 4 rows: L, L, R, L
    visual = ''.join(['L' if t == 1 else 'R' for t in example_partial])
    print(f"\nPartial game: {visual} (4 rows complete)")
    
    pred, conf, num_matches = analyzer.predict_next_tile_from_patterns(example_partial)
    
    if num_matches > 0:
        pred_visual = 'Left' if pred == 1 else 'Right'
        print(f"Prediction for row 5: {pred_visual}")
        print(f"Confidence: {conf*100:.1f}%")
        print(f"Based on {num_matches} matching historical patterns")
    else:
        print("No historical matches - would fall back to ML prediction")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
