#!/usr/bin/env python3
"""
Dragon Tower Live Game Predictor
Interactive tool for making predictions during gameplay
"""

import sys
from pathlib import Path

try:
    from dragon_tower_predictor import DragonTowerPredictor
    from pattern_analyzer import PatternAnalyzer
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages: pip install pandas numpy scikit-learn joblib")
    sys.exit(1)

class GamePredictor:
    """Interactive game prediction assistant"""
    
    def __init__(self):
        self.predictor = None
        self.pattern_analyzer = None
        self.game_history = []
        self.load_stats()
        self.load_pattern_analyzer()
        
    def load_stats(self):
        """Load historical statistics from data"""
        try:
            df = pd.read_csv('dragon_tower_logs/combined.csv')
            self.row_stats = {}
            for row in range(1, 10):
                row_data = df[df['ROW'] == row]
                left_prob = row_data['LeftTile'].mean()
                self.row_stats[row] = {
                    'left_prob': left_prob,
                    'right_prob': 1 - left_prob,
                    'total_games': len(row_data)
                }
        except Exception as e:
            print(f"Warning: Could not load historical stats: {e}")
            self.row_stats = {}
    
    def load_model(self):
        """Load trained ML model"""
        try:
            self.predictor = DragonTowerPredictor.load_models('dragon_tower_model.pkl')
            print("✓ ML model loaded successfully")
            return True
        except FileNotFoundError:
            print("⚠️  No trained model found. Run 'python dragon_tower_predictor.py' first.")
            print("   Falling back to statistical predictions only.")
            return False
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Falling back to statistical predictions only.")
            return False
    
    def load_pattern_analyzer(self):
        """Load pattern frequency data"""
        try:
            self.pattern_analyzer = PatternAnalyzer()
            self.pattern_analyzer.load_pattern_data('pattern_frequencies.json')
            print("✓ Pattern analyzer loaded successfully")
            return True
        except FileNotFoundError:
            print("⚠️  No pattern data found. Run 'python pattern_analyzer.py' first.")
            print("   Pattern-based predictions disabled.")
            return False
        except Exception as e:
            print(f"⚠️  Error loading pattern data: {e}")
            return False
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*60)
        print("🐉 DRAGON TOWER PREDICTION ASSISTANT 🐉")
        print("="*60)
        print("\nThis tool helps predict correct tiles during gameplay.")
        print("You'll input what you see, and it will suggest the next move.")
        print("\nLegend: L = Left tile | R = Right tile")
        print("="*60 + "\n")
    
    def get_prediction(self, row_number):
        """Get prediction for a specific row using hybrid approach"""
        prediction = {
            'ml_prediction': None,
            'ml_confidence': None,
            'pattern_prediction': None,
            'pattern_confidence': None,
            'pattern_matches': 0,
            'historical_left': None,
            'historical_right': None,
            'recommendation': None
        }
        
        # Get historical stats
        if row_number in self.row_stats:
            stats = self.row_stats[row_number]
            prediction['historical_left'] = stats['left_prob']
            prediction['historical_right'] = stats['right_prob']
        
        # Get pattern-based prediction if available
        if self.pattern_analyzer and row_number > 1:
            try:
                partial_pattern = tuple(self.game_history)
                pred, conf, matches = self.pattern_analyzer.predict_next_tile_from_patterns(partial_pattern)
                if matches > 0:
                    prediction['pattern_prediction'] = 'Left' if pred == 1 else 'Right'
                    prediction['pattern_confidence'] = conf
                    prediction['pattern_matches'] = matches
            except Exception as e:
                print(f"Warning: Pattern prediction failed: {e}")
        
        # Get ML prediction if available
        if self.predictor:
            try:
                pred, prob = self.predictor.predict_row(self.game_history, row_number)
                prediction['ml_prediction'] = 'Left' if pred == 1 else 'Right'
                prediction['ml_confidence'] = max(prob, 1 - prob)
            except Exception as e:
                print(f"Warning: ML prediction failed: {e}")
        
        # Make recommendation using hybrid approach
        # Priority: Pattern (if high confidence) > ML > Historical
        if prediction['pattern_prediction'] and prediction['pattern_confidence'] > 0.70:
            prediction['recommendation'] = prediction['pattern_prediction']
            prediction['reason'] = f"Pattern match ({prediction['pattern_confidence']*100:.1f}% confident, {prediction['pattern_matches']} matches)"
        elif prediction['ml_prediction'] and prediction['ml_confidence'] > 0.55:
            prediction['recommendation'] = prediction['ml_prediction']
            prediction['reason'] = f"ML model ({prediction['ml_confidence']*100:.1f}% confident)"
        elif prediction['pattern_prediction'] and prediction['pattern_confidence'] > 0.55:
            prediction['recommendation'] = prediction['pattern_prediction']
            prediction['reason'] = f"Pattern match ({prediction['pattern_confidence']*100:.1f}% confident)"
        elif prediction['historical_left']:
            if prediction['historical_left'] > 0.52:
                prediction['recommendation'] = 'Left'
                prediction['reason'] = f"Historical bias ({prediction['historical_left']*100:.1f}%)"
            elif prediction['historical_right'] > 0.52:
                prediction['recommendation'] = 'Right'
                prediction['reason'] = f"Historical bias ({prediction['historical_right']*100:.1f}%)"
            else:
                prediction['recommendation'] = 'Either'
                prediction['reason'] = '50/50 distribution'
        else:
            prediction['recommendation'] = 'Either'
            prediction['reason'] = 'Insufficient data'
        
        return prediction
    
    def display_prediction(self, row_number, prediction):
        """Display prediction in a user-friendly format"""
        print(f"\n📊 ROW {row_number} PREDICTION")
        print("-" * 50)
        
        # Historical stats
        if prediction['historical_left']:
            print(f"Historical Stats:")
            print(f"  Left:  {prediction['historical_left']*100:.1f}% ({'█' * int(prediction['historical_left']*20)})")
            print(f"  Right: {prediction['historical_right']*100:.1f}% ({'█' * int(prediction['historical_right']*20)})")
        
        # Pattern-based prediction
        if prediction['pattern_prediction']:
            confidence = prediction['pattern_confidence']
            bar_length = int(confidence * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"\nPattern Prediction: {prediction['pattern_prediction']}")
            print(f"Confidence: {confidence*100:.1f}% [{bar}]")
            print(f"Based on {prediction['pattern_matches']} matching patterns")
        
        # ML prediction
        if prediction['ml_prediction']:
            confidence = prediction['ml_confidence']
            bar_length = int(confidence * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"\nML Prediction: {prediction['ml_prediction']}")
            print(f"Confidence: {confidence*100:.1f}% [{bar}]")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION: {prediction['recommendation']}")
        print(f"   Reason: {prediction['reason']}")
        print("-" * 50)
    
    def play_game(self):
        """Interactive game session"""
        print("\n🎮 Starting new game...\n")
        self.game_history = []
        
        for row in range(1, 10):
            # Get and display prediction
            prediction = self.get_prediction(row)
            self.display_prediction(row, prediction)
            
            # Get user input - what tile was correct
            while True:
                choice = input(f"\nWhat was the CORRECT tile for Row {row}? (L/R or Q to quit): ").strip().upper()
                
                if choice == 'Q':
                    print("\n👋 Thanks for playing!")
                    return False
                elif choice in ['L', 'R']:
                    correct_tile = choice
                    break
                else:
                    print("Invalid input. Please enter L, R, or Q.")
            
            # Ask if they picked it correctly
            while True:
                picked = input(f"Did you PICK the correct tile ({correct_tile})? (Y/N): ").strip().upper()
                
                if picked == 'Y':
                    # They got it right, continue
                    if correct_tile == 'L':
                        self.game_history.append(1)  # Left = 1
                    else:
                        self.game_history.append(0)  # Right = 0
                    print(f"✓ Success! Moving to row {row + 1 if row < 9 else 'WIN'}")
                    break
                elif picked == 'N':
                    # They got it wrong - game over
                    print(f"\n💀 GAME OVER at Row {row}!")
                    
                    # Get the complete pattern for learning
                    complete_pattern = self.get_complete_pattern_on_failure(row, correct_tile)
                    
                    if complete_pattern:
                        self.save_learned_pattern(complete_pattern)
                        print("✓ Pattern recorded for future predictions!")
                    
                    print("\nStarting a new game...\n")
                    return self.play_game()  # Recursive call to start fresh
                else:
                    print("Invalid input. Please enter Y or N.")
        
        # Game complete - they won!
        print("\n" + "="*60)
        print("🎉 YOU WON! GAME COMPLETE!")
        print("="*60)
        print(f"\nYour winning sequence: {' '.join(['L' if t == 1 else 'R' for t in self.game_history])}")
        
        # Also save winning patterns
        self.save_learned_pattern(tuple(self.game_history))
        print("✓ Winning pattern recorded!")
        
        return True
    
    def get_complete_pattern_on_failure(self, failed_row, correct_tile):
        """Ask user to input the complete pattern shown after game over"""
        print(f"\n📝 The game revealed the complete pattern.")
        print(f"You already got rows 1-{failed_row-1}: {' '.join(['L' if t == 1 else 'R' for t in self.game_history])}")
        print(f"Row {failed_row} was: {correct_tile}")
        
        # Start with what we know
        complete = list(self.game_history)
        complete.append(1 if correct_tile == 'L' else 0)
        
        # Ask for remaining rows
        if failed_row < 9:
            print(f"\nPlease enter the remaining tiles (rows {failed_row+1}-9):")
            remaining_input = input(f"Enter {9 - failed_row} tiles as a sequence (e.g., LRLR or lrlr), or press Enter to skip: ").strip().upper()
            
            if remaining_input:
                # Validate input
                if len(remaining_input) == 9 - failed_row and all(c in ['L', 'R'] for c in remaining_input):
                    for char in remaining_input:
                        complete.append(1 if char == 'L' else 0)
                    return tuple(complete)
                else:
                    print("⚠️  Invalid input format. Skipping pattern recording.")
                    return None
            else:
                print("Skipping pattern recording.")
                return None
        else:
            # Failed on row 9, we have the complete pattern
            return tuple(complete)
    
    def save_learned_pattern(self, pattern):
        """Save a newly learned pattern to file for future analysis"""
        import json
        from datetime import datetime
        
        learned_file = 'learned_patterns.json'
        
        # Load existing data
        try:
            with open(learned_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {'patterns': [], 'pattern_counts': {}}
        
        # Convert pattern to string
        pattern_str = ''.join(['L' if t == 1 else 'R' for t in pattern])
        
        # Add to patterns list with timestamp
        data['patterns'].append({
            'pattern': pattern_str,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update counts
        if pattern_str not in data['pattern_counts']:
            data['pattern_counts'][pattern_str] = 0
        data['pattern_counts'][pattern_str] += 1
        
        # Save
        with open(learned_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update pattern analyzer if loaded
        if self.pattern_analyzer:
            # Add to pattern frequencies
            pattern_tuple = tuple([1 if c == 'L' else 0 for c in pattern_str])
            if pattern_tuple not in self.pattern_analyzer.pattern_frequencies:
                self.pattern_analyzer.pattern_frequencies[pattern_tuple] = 0
            self.pattern_analyzer.pattern_frequencies[pattern_tuple] += 1
            self.pattern_analyzer.total_games += 1
            
            # Save updated pattern data
            self.pattern_analyzer.save_pattern_data()
    
    def run(self):
        """Main application loop"""
        self.display_welcome()
        self.load_model()
        
        while True:
            print("\nOptions:")
            print("  1. Start new game prediction")
            print("  2. View row statistics")
            print("  3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                continue_playing = self.play_game()
                if not continue_playing:
                    break
                    
                # Ask if they want to play again
                again = input("\nPlay another game? (Y/N): ").strip().upper()
                if again != 'Y':
                    break
                    
            elif choice == '2':
                self.display_statistics()
                
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
    
    def display_statistics(self):
        """Display historical statistics"""
        print("\n" + "="*60)
        print("📈 HISTORICAL STATISTICS")
        print("="*60)
        
        if not self.row_stats:
            print("No statistics available.")
            return
        
        print("\nTile Distribution by Row:")
        print(f"{'Row':<6} {'Left %':<10} {'Right %':<10} {'Visual'}")
        print("-" * 60)
        
        for row in range(1, 10):
            if row in self.row_stats:
                stats = self.row_stats[row]
                left_pct = stats['left_prob'] * 100
                right_pct = stats['right_prob'] * 100
                
                # Create visual bar
                left_bar = '█' * int(stats['left_prob'] * 30)
                right_bar = '█' * int(stats['right_prob'] * 30)
                
                print(f"{row:<6} {left_pct:>6.1f}%    {right_pct:>6.1f}%    {left_bar}")
        
        print("\n" + "="*60)

def main():
    """Entry point"""
    predictor = GamePredictor()
    
    try:
        predictor.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
