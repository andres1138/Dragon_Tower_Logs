#!/usr/bin/env python3
"""
Dragon Tower ML Predictor
Trains and evaluates machine learning models to predict tile positions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class DragonTowerPredictor:
    """ML-based predictor for Dragon Tower tiles"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None
        self.row_accuracies = {}
        
    def load_data(self, filepath='dragon_tower_logs/combined.csv'):
        """Load and prepare data for training"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        
        # Prepare game sequences
        games = []
        for game_id in df['GAME'].unique():
            game_data = df[df['GAME'] == game_id].sort_values('ROW')
            tiles = game_data['LeftTile'].values
            games.append(tiles)
        
        self.games = np.array(games)
        print(f"Loaded {len(games)} games with {len(games[0])} rows each")
        return self.games
    
    def create_features(self, games, row_to_predict):
        """
        Create features for predicting a specific row
        Features include:
        - Previous row outcomes (if available)
        - Position in game (row number)
        - Pattern features (streaks, alternations)
        """
        X = []
        y = []
        
        for game in games:
            features = []
            
            # Add previous rows as features (if available)
            num_prev_rows = min(row_to_predict - 1, 3)  # Use up to 3 previous rows
            for i in range(num_prev_rows):
                features.append(game[row_to_predict - num_prev_rows + i - 1])
            
            # Pad with -1 if not enough previous rows
            while len(features) < 3:
                features.insert(0, -1)
            
            # Add row position as normalized feature
            features.append(row_to_predict / 9.0)
            
            # Add pattern features if we have previous rows
            if row_to_predict > 1:
                # Count consecutive same tiles before this row
                streak = 1
                for i in range(row_to_predict - 2, -1, -1):
                    if game[i] == game[row_to_predict - 2]:
                        streak += 1
                    else:
                        break
                features.append(min(streak / 5.0, 1.0))  # Normalized streak
                
                # Alternation pattern (0 or 1)
                if row_to_predict > 2:
                    alternating = int(game[row_to_predict - 2] != game[row_to_predict - 3])
                    features.append(alternating)
                else:
                    features.append(0)
            else:
                features.append(0)  # No streak for first row
                features.append(0)  # No alternation
            
            X.append(features)
            y.append(game[row_to_predict - 1])
        
        return np.array(X), np.array(y)
    
    def train_row_models(self, games):
        """Train separate models for each row"""
        print("\n" + "="*60)
        print("TRAINING ROW-SPECIFIC MODELS")
        print("="*60)
        
        all_accuracies = {
            'Random Forest': [],
            'Gradient Boosting': [],
            'Neural Network': []
        }
        
        for row in range(1, 10):
            print(f"\nTraining models for Row {row}...")
            
            # Create features for this row
            X, y = self.create_features(games, row)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            }
            
            row_accuracies = {}
            for name, model in models.items():
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                row_accuracies[name] = accuracy
                all_accuracies[name].append(accuracy)
                
                print(f"  {name}: {accuracy*100:.2f}%")
            
            # Store best model for this row
            best_name = max(row_accuracies, key=row_accuracies.get)
            self.models[row] = models[best_name]
            self.scalers[row] = scaler
            self.row_accuracies[row] = row_accuracies
        
        # Calculate and display average accuracies
        print("\n" + "="*60)
        print("AVERAGE ACCURACIES ACROSS ALL ROWS")
        print("="*60)
        for name in all_accuracies:
            avg_acc = np.mean(all_accuracies[name])
            std_acc = np.std(all_accuracies[name])
            print(f"  {name}: {avg_acc*100:.2f}% (±{std_acc*100:.2f}%)")
        
        # Select best overall approach
        best_overall = max(all_accuracies, key=lambda x: np.mean(all_accuracies[x]))
        self.best_model_name = best_overall
        print(f"\n  Best overall approach: {best_overall}")
        
        return all_accuracies
    
    def train_sequential_model(self, games):
        """Train a model that considers the entire game sequence"""
        print("\n" + "="*60)
        print("TRAINING SEQUENTIAL MODEL")
        print("="*60)
        print("(Using all rows together with sequence context)")
        
        X_all = []
        y_all = []
        row_labels = []
        
        # Create dataset with all rows
        for row in range(1, 10):
            X, y = self.create_features(games, row)
            X_all.extend(X)
            y_all.extend(y)
            row_labels.extend([row] * len(X))
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=6)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nSequential Model Accuracy: {accuracy*100:.2f}%")
        print(f"Baseline (random): 50%")
        print(f"Improvement over baseline: {(accuracy-0.5)*100:.2f}%")
        
        # Store as unified model
        self.models['unified'] = model
        self.scalers['unified'] = scaler
        
        return accuracy
    
    def predict_row(self, previous_tiles, row_number, use_unified=False):
        """
        Predict the next row's tile
        
        Args:
            previous_tiles: list of previous tile outcomes (1=left, 0=right)
            row_number: which row to predict (1-9)
            use_unified: whether to use unified model
            
        Returns:
            prediction (0 or 1), probability of left tile
        """
        # Create features
        features = []
        
        # Add previous rows (up to 3)
        num_prev = len(previous_tiles)
        for i in range(min(num_prev, 3)):
            features.append(previous_tiles[-(min(num_prev, 3)-i)])
        
        # Pad if needed
        while len(features) < 3:
            features.insert(0, -1)
        
        # Add row position
        features.append(row_number / 9.0)
        
        # Add streak feature
        if len(previous_tiles) > 0:
            streak = 1
            for i in range(len(previous_tiles) - 2, -1, -1):
                if previous_tiles[i] == previous_tiles[-1]:
                    streak += 1
                else:
                    break
            features.append(min(streak / 5.0, 1.0))
        else:
            features.append(0)
        
        # Add alternation feature
        if len(previous_tiles) >= 2:
            alternating = int(previous_tiles[-1] != previous_tiles[-2])
            features.append(alternating)
        else:
            features.append(0)
        
        features = np.array(features).reshape(1, -1)
        
        # Select model
        if use_unified and 'unified' in self.models:
            model = self.models['unified']
            scaler = self.scalers['unified']
        else:
            model = self.models.get(row_number)
            scaler = self.scalers.get(row_number)
        
        if model is None:
            # Fallback to 50/50
            return 1, 0.5
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            prob_left = proba[1] if len(proba) > 1 else 0.5
        else:
            prob_left = prediction  # Binary prediction
        
        return int(prediction), prob_left
    
    def save_models(self, filename='dragon_tower_model.pkl'):
        """Save trained models to file"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'best_model_name': self.best_model_name,
            'row_accuracies': self.row_accuracies
        }
        joblib.dump(model_data, filename)
        print(f"\n✓ Models saved to: {filename}")
    
    @classmethod
    def load_models(cls, filename='dragon_tower_model.pkl'):
        """Load trained models from file"""
        predictor = cls()
        model_data = joblib.load(filename)
        predictor.models = model_data['models']
        predictor.scalers = model_data['scalers']
        predictor.best_model_name = model_data['best_model_name']
        predictor.row_accuracies = model_data['row_accuracies']
        return predictor

def evaluate_prediction_value(games, predictor):
    """Evaluate if the predictions provide actual value"""
    print("\n" + "="*60)
    print("PREDICTION VALUE ANALYSIS")
    print("="*60)
    
    # Simulate using the predictor on test games
    test_games = games[-50:]  # Last 50 games for testing
    
    correct_predictions = 0
    total_predictions = 0
    confidence_bins = {'>60%': 0, '>70%': 0, '>80%': 0}
    confidence_correct = {'>60%': 0, '>70%': 0, '>80%': 0}
    
    for game in test_games:
        previous_tiles = []
        for row in range(1, 10):
            # Make prediction
            pred, prob = predictor.predict_row(previous_tiles, row)
            actual = game[row - 1]
            
            # Check accuracy
            if pred == actual:
                correct_predictions += 1
            total_predictions += 1
            
            # Track confidence
            confidence = max(prob, 1 - prob)
            if confidence > 0.6:
                confidence_bins['>60%'] += 1
                if pred == actual:
                    confidence_correct['>60%'] += 1
            if confidence > 0.7:
                confidence_bins['>70%'] += 1
                if pred == actual:
                    confidence_correct['>70%'] += 1
            if confidence > 0.8:
                confidence_bins['>80%'] += 1
                if pred == actual:
                    confidence_correct['>80%'] += 1
            
            # Add actual outcome to history
            previous_tiles.append(actual)
    
    overall_accuracy = correct_predictions / total_predictions
    
    print(f"\nOverall Prediction Accuracy: {overall_accuracy*100:.2f}%")
    print(f"Baseline (random guessing): 50%")
    print(f"Improvement: {(overall_accuracy - 0.5)*100:.2f}%")
    
    print("\nAccuracy by Confidence Level:")
    for level in ['>60%', '>70%', '>80%']:
        if confidence_bins[level] > 0:
            acc = confidence_correct[level] / confidence_bins[level]
            print(f"  {level}: {acc*100:.2f}% ({confidence_bins[level]} predictions)")
        else:
            print(f"  {level}: N/A (no predictions at this confidence)")
    
    # Conclusion
    print("\n" + "="*60)
    if overall_accuracy > 0.55:
        print("✓ MODEL PROVIDES VALUE")
        print(f"  Achieves {overall_accuracy*100:.1f}% accuracy (>{(overall_accuracy-0.5)*100:.1f}% above baseline)")
        print("  ✓ Patterns exist and are learnable!")
    elif overall_accuracy > 0.52:
        print("⚠️  MODEL PROVIDES MARGINAL VALUE")
        print(f"  Achieves {overall_accuracy*100:.1f}% accuracy (slight improvement)")
        print("  Patterns are weak but detectable")
    else:
        print("✗ MODEL PROVIDES MINIMAL VALUE")
        print("  Accuracy is near random baseline (50%)")
        print("  Game appears to use true random generation")

def main():
    """Main training pipeline"""
    print("="*60)
    print("DRAGON TOWER ML PREDICTOR")
    print("="*60)
    
    # Initialize predictor
    predictor = DragonTowerPredictor()
    
    # Load data
    games = predictor.load_data()
    
    # Train models
    predictor.train_row_models(games)
    predictor.train_sequential_model(games)
    
    # Evaluate prediction value
    evaluate_prediction_value(games, predictor)
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Use 'python predict_game.py' for live game predictions")
    print("  2. Check 'dragon_tower_model.pkl' for saved models")

if __name__ == "__main__":
    main()
