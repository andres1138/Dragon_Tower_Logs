#!/usr/bin/env python3
"""
Dragon Tower AI Predictor
=========================
An AI-assisted prediction tool for Dragon Tower (Hard mode: 2 columns, 9 rows).
Analyzes historical game patterns and provides real-time predictions during gameplay.

Usage:
    python3 dragon_tower_ai.py             # Interactive live predictor
    python3 dragon_tower_ai.py --analyze   # Full statistical analysis
    python3 dragon_tower_ai.py --backtest  # Backtest accuracy on historical data
"""

import csv
import os
import sys
import math
from collections import Counter, defaultdict
from datetime import datetime

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_ROWS = 9
NUM_PATTERNS = 2 ** NUM_ROWS  # 512
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dragon_tower_logs")
COMBINED_CSV = os.path.join(DATA_DIR, "combined.csv")

# ANSI color codes for terminal output
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_GREEN  = "\033[42m"
    BG_RED    = "\033[41m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_games(filepath=COMBINED_CSV):
    """
    Load all games from the combined CSV.
    Returns a list of tuples, each tuple is 9 elements of 0 (Left=egg) or 1 (Right=egg).
    """
    games = {}
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_data in reader:
            game_num = int(row_data["GAME"])
            row_num = int(row_data["ROW"])
            left = int(row_data["LeftTile"])
            # We store which side has the egg: 0 = Left, 1 = Right
            egg_side = 0 if left == 1 else 1
            if game_num not in games:
                games[game_num] = [None] * NUM_ROWS
            games[game_num][row_num - 1] = egg_side

    # Convert to sorted list of tuples
    result = []
    for game_num in sorted(games.keys()):
        game = games[game_num]
        if None not in game and len(game) == NUM_ROWS:
            result.append(tuple(game))
    return result


def load_all_individual_csvs():
    """Fallback: load from individual CSV files if combined.csv doesn't exist."""
    games = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith("dragon-tower-log-game") and fname.endswith(".csv"):
            filepath = os.path.join(DATA_DIR, fname)
            game = [None] * NUM_ROWS
            with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row_data in reader:
                    row_num = int(row_data["ROW"])
                    left = int(row_data["LeftTile"])
                    game[row_num - 1] = 0 if left == 1 else 1
            if None not in game:
                games.append(tuple(game))
    return games


def save_game_to_csv(game_sequence, games_count):
    """Save a new game to the combined CSV and as individual CSV."""
    game_num = games_count + 1

    # Append to combined.csv
    with open(COMBINED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row_idx, egg_side in enumerate(game_sequence):
            left = 1 if egg_side == 0 else 0
            right = 1 if egg_side == 1 else 0
            writer.writerow([game_num, row_idx + 1, left, right])

    # Also create individual CSV
    individual_path = os.path.join(DATA_DIR, f"dragon-tower-log-game{game_num}.csv")
    with open(individual_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GAME", "ROW", "LeftTile", "RightTile"])
        for row_idx, egg_side in enumerate(game_sequence):
            left = 1 if egg_side == 0 else 0
            right = 1 if egg_side == 1 else 0
            writer.writerow([game_num, row_idx + 1, left, right])

    return game_num


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────
class DragonTowerAnalyzer:
    def __init__(self, games):
        self.games = games
        self.num_games = len(games)
        self._compute_statistics()

    def _compute_statistics(self):
        """Pre-compute all statistics from game data."""
        # Row-level bias: probability of Left (0) for each row
        self.row_bias = [0.0] * NUM_ROWS
        for row in range(NUM_ROWS):
            left_count = sum(1 for g in self.games if g[row] == 0)
            self.row_bias[row] = left_count / self.num_games if self.num_games > 0 else 0.5

        # Transition probabilities: P(next | current)
        # transition_probs[current_side][row] = P(next_row = 0 | current_row = current_side)
        self.transition_counts = [[Counter(), Counter()] for _ in range(NUM_ROWS - 1)]
        for g in self.games:
            for row in range(NUM_ROWS - 1):
                current = g[row]
                next_val = g[row + 1]
                self.transition_counts[row][current][next_val] += 1

        # Full pattern frequency
        self.pattern_counts = Counter(self.games)

        # Prefix -> continuation mapping (for conditional probabilities)
        self.prefix_map = defaultdict(list)
        for g in self.games:
            for length in range(1, NUM_ROWS + 1):
                prefix = g[:length]
                self.prefix_map[prefix].append(g)

    def get_row_bias(self, row):
        """P(Left) for a given row."""
        return self.row_bias[row]

    def get_transition_prob(self, row, current_side):
        """P(next_row = Left | current_row = current_side) for row transition."""
        if row >= NUM_ROWS - 1:
            return 0.5
        counts = self.transition_counts[row][current_side]
        total = counts[0] + counts[1]
        if total == 0:
            return 0.5
        return counts[0] / total

    def get_conditional_prob(self, prefix):
        """
        Given a prefix (rows revealed so far), compute P(next = Left) based on
        all historical games that share this prefix.
        Returns (prob_left, num_matching_games, matching_games).
        """
        prefix_tuple = tuple(prefix)
        next_row = len(prefix)

        if next_row >= NUM_ROWS:
            return 0.5, 0, []

        # Find all games matching this prefix
        matching = [g for g in self.games if g[:len(prefix)] == prefix_tuple]

        if not matching:
            return 0.5, 0, []

        left_count = sum(1 for g in matching if g[next_row] == 0)
        prob_left = left_count / len(matching)
        return prob_left, len(matching), matching

    def get_top_patterns_for_prefix(self, prefix, top_n=5):
        """
        Given a prefix, return the top N most likely full patterns.
        """
        prefix_tuple = tuple(prefix)
        matching = [g for g in self.games if g[:len(prefix)] == prefix_tuple]
        if not matching:
            return []
        pattern_freq = Counter(matching)
        return pattern_freq.most_common(top_n)

    def get_fused_prediction(self, prefix):
        """
        Multi-signal fusion prediction for the next row.
        Combines: prefix matching, row bias, transition probability, pattern weighting.
        Returns (prob_left, confidence, signals_dict).
        """
        next_row = len(prefix)
        if next_row >= NUM_ROWS:
            return 0.5, 0.0, {}

        signals = {}
        weights = {}

        # Signal 1: Conditional probability from prefix matching (STRONGEST)
        cond_prob, num_matches, _ = self.get_conditional_prob(prefix)
        signals["prefix_match"] = cond_prob
        # Weight by number of matches (more matches = more reliable)
        # Minimum weight of 1, scales up with matches
        if num_matches > 0:
            weights["prefix_match"] = min(5.0, 1.0 + math.log2(max(1, num_matches)))
        else:
            weights["prefix_match"] = 0.0

        # Signal 2: Row bias
        row_prob = self.get_row_bias(next_row)
        signals["row_bias"] = row_prob
        weights["row_bias"] = 1.0

        # Signal 3: Transition probability (if we have at least one row)
        if len(prefix) > 0:
            trans_prob = self.get_transition_prob(next_row - 1, prefix[-1])
            signals["transition"] = trans_prob
            weights["transition"] = 1.5
        else:
            signals["transition"] = 0.5
            weights["transition"] = 0.0

        # Signal 4: Pattern frequency weighting
        # Check if remaining possible patterns lean one way
        if num_matches >= 3:
            top_patterns = self.get_top_patterns_for_prefix(prefix, top_n=10)
            pattern_left = 0
            pattern_total = 0
            for pattern, count in top_patterns:
                pattern_left += count if pattern[next_row] == 0 else 0
                pattern_total += count
            if pattern_total > 0:
                signals["pattern_freq"] = pattern_left / pattern_total
                weights["pattern_freq"] = 1.2
            else:
                signals["pattern_freq"] = 0.5
                weights["pattern_freq"] = 0.0
        else:
            signals["pattern_freq"] = 0.5
            weights["pattern_freq"] = 0.0

        # Weighted average
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.5, 0.0, signals

        fused_prob = sum(signals[k] * weights[k] for k in signals) / total_weight

        # Confidence = how far from 50/50 we are, scaled by data quality
        raw_confidence = abs(fused_prob - 0.5) * 2  # 0 to 1
        data_quality = min(1.0, num_matches / 20) if num_matches > 0 else 0.3
        confidence = raw_confidence * data_quality

        detail = {k: {"prob_left": signals[k], "weight": weights[k]} for k in signals}
        detail["num_matches"] = num_matches
        detail["fused_prob_left"] = fused_prob
        detail["confidence"] = confidence

        return fused_prob, confidence, detail


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(text):
    width = 60
    print(f"\n{C.BOLD}{C.CYAN}{'═' * width}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  🐉 {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * width}{C.RESET}")


def print_subheader(text):
    print(f"\n{C.BOLD}{C.YELLOW}  ── {text} ──{C.RESET}")


def print_separator():
    print(f"{C.DIM}{'─' * 60}{C.RESET}")


def format_pct(value):
    """Format a probability as a colored percentage."""
    pct = value * 100
    if pct >= 60:
        color = C.GREEN
    elif pct >= 55:
        color = C.YELLOW
    elif pct <= 40:
        color = C.RED
    elif pct <= 45:
        color = C.YELLOW
    else:
        color = C.WHITE
    return f"{color}{pct:.1f}%{C.RESET}"


def format_side(side, prob=None):
    """Format LEFT or RIGHT with color."""
    if side == 0:
        label = "⬅  LEFT"
        color = C.CYAN
    else:
        label = "RIGHT ➡"
        color = C.MAGENTA
    if prob is not None:
        return f"{C.BOLD}{color}{label}{C.RESET} ({format_pct(prob)})"
    return f"{C.BOLD}{color}{label}{C.RESET}"


def format_confidence_bar(confidence):
    """Create a visual confidence bar."""
    filled = int(confidence * 20)
    empty = 20 - filled
    if confidence >= 0.3:
        color = C.GREEN
    elif confidence >= 0.15:
        color = C.YELLOW
    else:
        color = C.RED
    return f"{color}{'█' * filled}{'░' * empty}{C.RESET} {confidence * 100:.0f}%"


def display_tower(prefix, predictions=None):
    """Display the current tower state with known and predicted rows."""
    print(f"\n  {C.BOLD}{'TOWER STATE':^40}{C.RESET}")
    print(f"  {'─' * 40}")

    for row in range(NUM_ROWS - 1, -1, -1):
        row_label = f"Row {row + 1}"
        if row < len(prefix):
            # Known row
            side = prefix[row]
            if side == 0:
                left = f"{C.GREEN}{C.BOLD} 🥚 {C.RESET}"
                right = f"{C.RED} 💀 {C.RESET}"
            else:
                left = f"{C.RED} 💀 {C.RESET}"
                right = f"{C.GREEN}{C.BOLD} 🥚 {C.RESET}"
            print(f"  {C.DIM}{row_label:>6}{C.RESET}  │{left}│{right}│  {C.GREEN}✓{C.RESET}")
        elif row == len(prefix) and predictions:
            # Next row to predict
            prob_left = predictions.get("prob_left", 0.5)
            if prob_left >= 0.5:
                left = f"{C.YELLOW}{C.BOLD} ❓  {C.RESET}"
                right = f"{C.DIM} ❓  {C.RESET}"
            else:
                left = f"{C.DIM} ❓  {C.RESET}"
                right = f"{C.YELLOW}{C.BOLD} ❓  {C.RESET}"
            print(f"  {C.BOLD}{C.YELLOW}{row_label:>6}{C.RESET}  │{left}│{right}│  {C.YELLOW}◀ NEXT{C.RESET}")
        else:
            # Unknown future row
            left = f"{C.DIM} · · {C.RESET}"
            right = f"{C.DIM} · · {C.RESET}"
            print(f"  {C.DIM}{row_label:>6}{C.RESET}  │{left}│{right}│")

    print(f"  {'─' * 40}")
    print(f"  {'':>6}   LEFT  RIGHT")


def display_prediction(analyzer, prefix):
    """Display the AI's prediction for the next row."""
    next_row = len(prefix)
    if next_row >= NUM_ROWS:
        print(f"\n  {C.GREEN}{C.BOLD}🎉 All 9 rows revealed! Full pattern complete.{C.RESET}")
        return

    prob_left, confidence, detail = analyzer.get_fused_prediction(prefix)
    num_matches = detail.get("num_matches", 0)

    print_subheader(f"PREDICTION FOR ROW {next_row + 1}")

    # Main recommendation
    if prob_left >= 0.5:
        recommended = 0
        rec_prob = prob_left
    else:
        recommended = 1
        rec_prob = 1 - prob_left

    print(f"\n  {C.BOLD}Recommendation: {format_side(recommended, rec_prob)}{C.RESET}")
    print(f"  Confidence:    {format_confidence_bar(confidence)}")
    print(f"  Matching games: {C.BOLD}{num_matches}{C.RESET} / {analyzer.num_games}")

    # Signal breakdown
    print(f"\n  {C.DIM}Signal breakdown:{C.RESET}")
    for signal_name in ["prefix_match", "row_bias", "transition", "pattern_freq"]:
        if signal_name in detail and isinstance(detail[signal_name], dict):
            info = detail[signal_name]
            prob = info["prob_left"]
            weight = info["weight"]
            side = "LEFT" if prob >= 0.5 else "RIGHT"
            dom_prob = prob if prob >= 0.5 else (1 - prob)
            bar_len = int(weight * 4)
            weight_bar = "●" * bar_len + "○" * (20 - bar_len)
            label = signal_name.replace("_", " ").title()
            print(f"    {label:>16}: {side:>5} {dom_prob * 100:5.1f}%  weight: {C.DIM}{weight_bar[:5]}{C.RESET}")

    # Top matching patterns
    if num_matches > 0:
        top = analyzer.get_top_patterns_for_prefix(prefix, top_n=5)
        if top:
            print(f"\n  {C.DIM}Top matching patterns (L=Left egg, R=Right egg):{C.RESET}")
            for pattern, count in top:
                display = ""
                for i, s in enumerate(pattern):
                    if i < len(prefix):
                        display += f"{C.GREEN}{'L' if s == 0 else 'R'}{C.RESET}"
                    elif i == len(prefix):
                        display += f"{C.YELLOW}{C.BOLD}{'L' if s == 0 else 'R'}{C.RESET}"
                    else:
                        display += f"{C.DIM}{'L' if s == 0 else 'R'}{C.RESET}"
                freq = count / analyzer.num_games * 100
                print(f"    {display}  (seen {count}x, {freq:.1f}%)")


# ─────────────────────────────────────────────
# STATISTICAL ANALYSIS MODE
# ─────────────────────────────────────────────
def run_analysis(games):
    """Full statistical analysis of all game data."""
    analyzer = DragonTowerAnalyzer(games)

    print_header("DRAGON TOWER STATISTICAL ANALYSIS")
    print(f"\n  Total games loaded: {C.BOLD}{analyzer.num_games}{C.RESET}")
    print(f"  Possible patterns:  {C.BOLD}{NUM_PATTERNS}{C.RESET}")
    print(f"  Unique patterns seen: {C.BOLD}{len(analyzer.pattern_counts)}{C.RESET}")
    coverage = len(analyzer.pattern_counts) / NUM_PATTERNS * 100
    print(f"  Pattern coverage:   {C.BOLD}{coverage:.1f}%{C.RESET}")

    # Row-level bias
    print_subheader("ROW-LEVEL BIAS (% Left)")
    expected = 50.0
    for row in range(NUM_ROWS):
        pct = analyzer.row_bias[row] * 100
        deviation = pct - expected
        bar_left = int(pct / 2.5)
        bar_right = 40 - bar_left
        dev_color = C.GREEN if abs(deviation) > 3 else C.DIM
        print(f"  Row {row + 1}: {C.CYAN}{'█' * bar_left}{C.MAGENTA}{'█' * bar_right}{C.RESET}"
              f"  L:{pct:5.1f}% R:{100-pct:5.1f}%  {dev_color}({deviation:+.1f}%){C.RESET}")

    # Transition probabilities
    print_subheader("TRANSITION PROBABILITIES")
    print(f"  {'':>10} {'After LEFT →':>20} {'After RIGHT →':>20}")
    for row in range(NUM_ROWS - 1):
        p_left_after_left = analyzer.get_transition_prob(row, 0)
        p_left_after_right = analyzer.get_transition_prob(row, 1)
        print(f"  Row {row+1}→{row+2}:"
              f"  L:{p_left_after_left*100:5.1f}% R:{(1-p_left_after_left)*100:5.1f}%"
              f"    L:{p_left_after_right*100:5.1f}% R:{(1-p_left_after_right)*100:5.1f}%")

    # Streak analysis
    print_subheader("STREAK ANALYSIS")
    streak_counts = Counter()
    for g in games:
        streak = 1
        for i in range(1, NUM_ROWS):
            if g[i] == g[i-1]:
                streak += 1
            else:
                streak_counts[streak] += 1
                streak = 1
        streak_counts[streak] += 1

    total_streaks = sum(streak_counts.values())
    for length in sorted(streak_counts.keys()):
        count = streak_counts[length]
        pct = count / total_streaks * 100
        bar = "█" * int(pct / 2)
        print(f"  Streak {length}: {count:>4}x ({pct:5.1f}%) {C.CYAN}{bar}{C.RESET}")

    # Most common patterns
    print_subheader("TOP 15 MOST COMMON PATTERNS")
    expected_freq = analyzer.num_games / NUM_PATTERNS
    for i, (pattern, count) in enumerate(analyzer.pattern_counts.most_common(15)):
        display = " ".join("L" if s == 0 else "R" for s in pattern)
        ratio = count / expected_freq
        ratio_color = C.GREEN if ratio > 2 else (C.YELLOW if ratio > 1.5 else C.WHITE)
        print(f"  {i+1:>2}. {display}  "
              f"× {count:>2}  ({ratio_color}{ratio:.1f}x expected{C.RESET})")

    # Patterns that appear more than expected
    print_subheader("PATTERNS APPEARING 3+ TIMES")
    hot_patterns = [(p, c) for p, c in analyzer.pattern_counts.items() if c >= 3]
    hot_patterns.sort(key=lambda x: -x[1])
    if hot_patterns:
        for pattern, count in hot_patterns:
            display = " ".join("L" if s == 0 else "R" for s in pattern)
            print(f"  {display}  × {count} {C.GREEN}🔥{C.RESET}")
    else:
        print(f"  {C.DIM}No patterns appear 3 or more times.{C.RESET}")

    # First 5 rows analysis (the money zone)
    print_subheader("FIRST 5 ROWS ANALYSIS (YOUR MONEY ZONE)")
    prefix_5_counts = Counter(g[:5] for g in games)
    total_5_patterns = 2 ** 5  # 32
    print(f"  Unique 5-row prefixes seen: {C.BOLD}{len(prefix_5_counts)}{C.RESET} / {total_5_patterns}")
    print(f"\n  Top 10 most common 5-row openings:")
    for i, (prefix, count) in enumerate(prefix_5_counts.most_common(10)):
        display = " ".join("L" if s == 0 else "R" for s in prefix)
        pct = count / analyzer.num_games * 100
        print(f"  {i+1:>2}. {display}  × {count:>2} ({pct:.1f}%)")

    print()


# ─────────────────────────────────────────────
# BACKTESTING MODE
# ─────────────────────────────────────────────
def run_backtest(games):
    """Leave-one-out backtesting: for each game, predict using all other games."""
    print_header("DRAGON TOWER BACKTEST")
    print(f"\n  Running leave-one-out backtest on {len(games)} games...")
    print(f"  {C.DIM}(Using all other games to predict each game){C.RESET}\n")

    row_correct = [0] * NUM_ROWS
    row_total = [0] * NUM_ROWS
    first_n_correct = Counter()  # first_n_correct[n] = games where rows 1..n were all correct
    total_games = len(games)

    for i, target_game in enumerate(games):
        # Build training set: all games except the target
        training = games[:i] + games[i+1:]
        analyzer = DragonTowerAnalyzer(training)

        # Simulate playing row by row
        prefix = []
        all_correct_so_far = True
        for row in range(NUM_ROWS):
            prob_left, _, _ = analyzer.get_fused_prediction(prefix)

            # AI picks the side with higher probability
            predicted = 0 if prob_left >= 0.5 else 1
            actual = target_game[row]

            row_total[row] += 1
            if predicted == actual:
                row_correct[row] += 1
            else:
                all_correct_so_far = False

            if all_correct_so_far:
                first_n_correct[row + 1] = first_n_correct.get(row + 1, 0) + 1

            # Reveal the actual result for next prediction
            prefix.append(actual)

        # Progress indicator
        if (i + 1) % 50 == 0 or i + 1 == total_games:
            pct = (i + 1) / total_games * 100
            print(f"\r  Progress: {pct:.0f}% ({i+1}/{total_games})", end="", flush=True)

    print("\n")

    # Results
    print_subheader("PER-ROW ACCURACY")
    print(f"  {'Row':>6}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>9}  {'vs Random':>10}")
    print_separator()
    for row in range(NUM_ROWS):
        acc = row_correct[row] / row_total[row] * 100 if row_total[row] > 0 else 0
        vs_random = acc - 50
        color = C.GREEN if vs_random > 2 else (C.YELLOW if vs_random > 0 else C.RED)
        bar = "█" * int(acc / 2.5)
        print(f"  Row {row+1}:  {row_correct[row]:>5}  / {row_total[row]:>5}  "
              f"{color}{acc:6.1f}%{C.RESET}   {color}{vs_random:+5.1f}%{C.RESET}  {color}{bar}{C.RESET}")

    # First N rows correct
    print_subheader("CONSECUTIVE CORRECT FROM ROW 1")
    print(f"  {'Rows':>12}  {'Games':>6}  {'Rate':>8}  {'vs Random':>10}")
    print_separator()
    for n in range(1, NUM_ROWS + 1):
        count = first_n_correct.get(n, 0)
        rate = count / total_games * 100
        random_rate = (1/2)**n * 100
        improvement = rate - random_rate
        color = C.GREEN if improvement > 0.5 else (C.YELLOW if improvement > 0 else C.RED)
        print(f"  Rows 1-{n}:  {count:>5}  {rate:6.1f}%  "
              f"random: {random_rate:5.1f}%  {color}{improvement:+5.1f}%{C.RESET}")

    print(f"\n  {C.BOLD}Key insight:{C.RESET}")
    r5_rate = first_n_correct.get(5, 0) / total_games * 100
    r5_random = (1/2)**5 * 100
    if r5_rate > r5_random:
        print(f"  Getting rows 1-5 correct: {C.GREEN}{C.BOLD}{r5_rate:.1f}%{C.RESET}"
              f" vs {r5_random:.1f}% random = {C.GREEN}{C.BOLD}{r5_rate - r5_random:+.1f}% edge!{C.RESET}")
    else:
        print(f"  Getting rows 1-5 correct: {C.YELLOW}{r5_rate:.1f}%{C.RESET}"
              f" vs {r5_random:.1f}% random. Data may be truly random.")

    print()


# ─────────────────────────────────────────────
# INTERACTIVE LIVE PREDICTOR
# ─────────────────────────────────────────────
def run_interactive(games):
    """Live interactive predictor for Dragon Tower gameplay."""
    analyzer = DragonTowerAnalyzer(games)

    while True:
        clear_screen()
        print_header("DRAGON TOWER AI PREDICTOR")
        print(f"\n  {C.DIM}Games in database: {analyzer.num_games}{C.RESET}")
        print(f"\n  {C.BOLD}Choose an option:{C.RESET}")
        print(f"    {C.CYAN}1{C.RESET}) 🎮  Start new game (live prediction)")
        print(f"    {C.CYAN}2{C.RESET}) 📊  View statistics")
        print(f"    {C.CYAN}3{C.RESET}) 🧪  Run backtest")
        print(f"    {C.CYAN}4{C.RESET}) 🏆  View hot patterns")
        print(f"    {C.CYAN}5{C.RESET}) 🚪  Exit")
        print()

        choice = input(f"  {C.BOLD}Enter choice (1-5): {C.RESET}").strip()

        if choice == "1":
            run_game_session(analyzer, games)
        elif choice == "2":
            run_analysis(games)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "3":
            run_backtest(games)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "4":
            show_hot_patterns(analyzer)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "5":
            print(f"\n  {C.CYAN}Good luck on the tower! 🐉{C.RESET}\n")
            break
        else:
            print(f"  {C.RED}Invalid choice.{C.RESET}")
            input(f"  {C.DIM}Press Enter to continue...{C.RESET}")


def run_game_session(analyzer, games):
    """Run a single game prediction session."""
    prefix = []
    game_active = True

    while game_active:
        clear_screen()
        print_header(f"LIVE GAME — Row {len(prefix) + 1} of {NUM_ROWS}")

        # Show current tower
        predictions = {}
        if len(prefix) < NUM_ROWS:
            prob_left, _, _ = analyzer.get_fused_prediction(prefix)
            predictions["prob_left"] = prob_left
        display_tower(prefix, predictions)

        # Show prediction
        if len(prefix) < NUM_ROWS:
            display_prediction(analyzer, prefix)

        if len(prefix) >= NUM_ROWS:
            print(f"\n  {C.GREEN}{C.BOLD}🎉 All 9 rows complete!{C.RESET}")
            pattern_display = " ".join("L" if s == 0 else "R" for s in prefix)
            print(f"  Pattern: {pattern_display}")

            # Check if pattern was seen before
            count = analyzer.pattern_counts.get(tuple(prefix), 0)
            if count > 0:
                print(f"  {C.YELLOW}This pattern was seen {count}x before!{C.RESET}")
            else:
                print(f"  {C.CYAN}New pattern! Not seen in the database.{C.RESET}")

            # Offer to save
            print(f"\n  {C.BOLD}Options:{C.RESET}")
            print(f"    {C.CYAN}S{C.RESET}) Save this game to database")
            print(f"    {C.CYAN}N{C.RESET}) New game (don't save)")
            print(f"    {C.CYAN}Q{C.RESET}) Back to menu")
            choice = input(f"\n  {C.BOLD}Choice: {C.RESET}").strip().upper()

            if choice == "S":
                game_num = save_game_to_csv(prefix, len(games))
                games.append(tuple(prefix))
                analyzer = DragonTowerAnalyzer(games)
                print(f"  {C.GREEN}✓ Saved as game #{game_num}{C.RESET}")
                input(f"  {C.DIM}Press Enter to continue...{C.RESET}")
                prefix = []
            elif choice == "N":
                prefix = []
            else:
                game_active = False
            continue

        # Get user input for current row
        print(f"\n  {C.BOLD}What was Row {len(prefix) + 1}?{C.RESET}")
        print(f"    {C.CYAN}L{C.RESET}) Left had the egg")
        print(f"    {C.CYAN}R{C.RESET}) Right had the egg")
        print(f"    {C.CYAN}U{C.RESET}) Undo last row")
        print(f"    {C.CYAN}Q{C.RESET}) Quit game (back to menu)")

        result = input(f"\n  {C.BOLD}Enter (L/R/U/Q): {C.RESET}").strip().upper()

        if result == "L":
            prefix.append(0)
        elif result == "R":
            prefix.append(1)
        elif result == "U":
            if prefix:
                prefix.pop()
                print(f"  {C.YELLOW}↩ Undid last row.{C.RESET}")
            else:
                print(f"  {C.RED}Nothing to undo.{C.RESET}")
        elif result == "Q":
            # Offer to save partial game? No - we only save complete games
            if len(prefix) > 0:
                print(f"\n  {C.YELLOW}Note: Partial games are not saved.{C.RESET}")
                print(f"  Enter the remaining rows to save, or press Q again to quit.")
                confirm = input(f"  {C.BOLD}Quit? (Y/N): {C.RESET}").strip().upper()
                if confirm == "Y":
                    game_active = False
            else:
                game_active = False
        else:
            print(f"  {C.RED}Invalid input. Enter L, R, U, or Q.{C.RESET}")
            input(f"  {C.DIM}Press Enter to continue...{C.RESET}")


def show_hot_patterns(analyzer):
    """Show patterns that appear frequently — 'hot' patterns."""
    print_header("HOT PATTERNS")

    # Full 9-row hot patterns
    print_subheader("FULL PATTERNS (appearing 2+ times)")
    hot = [(p, c) for p, c in analyzer.pattern_counts.items() if c >= 2]
    hot.sort(key=lambda x: -x[1])
    if hot:
        for pattern, count in hot:
            display = " ".join("L" if s == 0 else "R" for s in pattern)
            fire = "🔥" * min(count, 5)
            print(f"  {display}  × {count} {fire}")
    else:
        print(f"  {C.DIM}No full patterns appear more than once.{C.RESET}")

    # 5-row prefix hot patterns
    print_subheader("5-ROW OPENINGS (appearing 5+ times)")
    prefix_5 = Counter(g[:5] for g in analyzer.games)
    hot_5 = [(p, c) for p, c in prefix_5.items() if c >= 5]
    hot_5.sort(key=lambda x: -x[1])
    if hot_5:
        for prefix, count in hot_5:
            display = " ".join("L" if s == 0 else "R" for s in prefix)
            pct = count / analyzer.num_games * 100
            expected_pct = 100 / 32  # 3.125%
            ratio = pct / expected_pct
            print(f"  {display}  × {count:>2} ({pct:.1f}%, {ratio:.1f}x expected)")
    else:
        print(f"  {C.DIM}No 5-row openings appear 5 or more times.{C.RESET}")

    # 3-row prefix hot patterns (most useful for early prediction)
    print_subheader("3-ROW OPENINGS (appearing 15+ times)")
    prefix_3 = Counter(g[:3] for g in analyzer.games)
    hot_3 = [(p, c) for p, c in prefix_3.items() if c >= 15]
    hot_3.sort(key=lambda x: -x[1])
    if hot_3:
        for prefix, count in hot_3:
            display = " ".join("L" if s == 0 else "R" for s in prefix)
            pct = count / analyzer.num_games * 100
            expected_pct = 100 / 8  # 12.5%
            ratio = pct / expected_pct
            print(f"  {display}  × {count:>2} ({pct:.1f}%, {ratio:.1f}x expected)")
    else:
        print(f"  {C.DIM}No 3-row openings appear 15 or more times. Lowering threshold...{C.RESET}")
        hot_3 = [(p, c) for p, c in prefix_3.items() if c >= 10]
        hot_3.sort(key=lambda x: -x[1])
        for prefix, count in hot_3[:10]:
            display = " ".join("L" if s == 0 else "R" for s in prefix)
            pct = count / analyzer.num_games * 100
            print(f"  {display}  × {count:>2} ({pct:.1f}%)")

    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load data
    print(f"{C.DIM}Loading game data...{C.RESET}")
    if os.path.exists(COMBINED_CSV):
        games = load_games(COMBINED_CSV)
    else:
        games = load_all_individual_csvs()

    if not games:
        print(f"{C.RED}Error: No games found in {DATA_DIR}{C.RESET}")
        sys.exit(1)

    print(f"{C.GREEN}Loaded {len(games)} games.{C.RESET}")

    # Check for command-line arguments
    if len(sys.argv) > 1:
        if "--analyze" in sys.argv:
            run_analysis(games)
        elif "--backtest" in sys.argv:
            run_backtest(games)
        elif "--help" in sys.argv:
            print(__doc__)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print(f"Usage: python3 {sys.argv[0]} [--analyze | --backtest | --help]")
    else:
        run_interactive(games)


if __name__ == "__main__":
    main()
