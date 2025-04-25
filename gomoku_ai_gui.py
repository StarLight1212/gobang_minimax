# -*- coding: utf-8 -*-
import tkinter as tk
import tkinter.messagebox
import time
import math
import random
from collections import deque
import sys

# Increase recursion depth limit for potentially deep searches
# Be cautious with this, as very deep searches can consume significant resources
try:
    # Setting a reasonable limit, adjust if needed and if your system allows
    sys.setrecursionlimit(3000)
except Exception as e:
    print(f"Warning: Could not set recursion depth limit. Default limit applies. Error: {e}")

# --- Configuration (config.py equivalent) ---
config = {
    "enableCache": True,
    "pointsLimit": 20,  # Max nodes to search per layer (used in move generation)
    "depth": 4,  # AI search depth (adjust for difficulty)
    "board_size": 15,
    # 'onlyInLine', 'inlineCount', 'inLineDistance' are complex optimizations.
    # We'll omit them for this initial Python version for simplicity,
    # but they could be added later. Focus on core minimax first.
}

# --- Shapes (shape.py equivalent) ---
shapes = {
    "FIVE": 5,
    "BLOCK_FIVE": 50,  # Technically a win, but used in eval sometimes
    "FOUR": 4,
    "FOUR_FOUR": 44,  # Double four
    "FOUR_THREE": 43,  # Four-three
    "THREE_THREE": 33,  # Double three
    "BLOCK_FOUR": 40,
    "THREE": 3,
    "BLOCK_THREE": 30,
    "TWO_TWO": 22,  # Double two
    "TWO": 2,
    "BLOCK_TWO": 20,  # Added for completeness, not in original getRealShapeScore
    "ONE": 1,  # Added for completeness
    "BLOCK_ONE": 10,  # Added for completeness
    "NONE": 0,
}


def is_five(shape):
    return shape == shapes["FIVE"] or shape == shapes["BLOCK_FIVE"]


def is_four(shape):
    # In some contexts, FIVE might be treated as a FOUR for scoring before winning
    return shape == shapes["FOUR"] or shape == shapes["BLOCK_FOUR"] or is_five(shape)


def count_shape(board, x, y, dx, dy, role, size):
    opponent = -role
    inner_empty_count = 0
    temp_empty_count = 0
    self_count = 0
    total_length = 0
    side_empty_count = 0
    no_empty_self_count = 0
    one_empty_self_count = 0

    # Search in the given direction (dx, dy)
    for i in range(1, 6):  # Check up to 5 steps away
        nx, ny = x + i * dx, y + i * dy

        # Check bounds (adjusting for the border)
        if not (0 <= nx < size and 0 <= ny < size):
            break  # Wall hit equivalent

        current_role = board[nx + 1][ny + 1]

        if current_role == 2 or current_role == opponent:  # Boundary or opponent
            break

        total_length += 1
        if current_role == role:
            self_count += 1
            side_empty_count = 0  # Reset side empty count on finding own piece
            if temp_empty_count > 0:
                inner_empty_count += temp_empty_count
                temp_empty_count = 0

            if inner_empty_count == 0:
                no_empty_self_count += 1
                one_empty_self_count += 1
            elif inner_empty_count == 1:
                one_empty_self_count += 1

        elif current_role == 0:  # Empty spot
            temp_empty_count += 1
            side_empty_count += 1

        # If we have two consecutive empty spots at the edge, stop extending
        if side_empty_count >= 2:
            break

    # If no inner empty spaces were found, one_empty_self_count should reflect only contiguous pieces
    if inner_empty_count == 0:
        one_empty_self_count = 0  # Reset if only contiguous pieces found

    return {
        "self_count": self_count,
        "total_length": total_length,
        "no_empty_self_count": no_empty_self_count,
        "one_empty_self_count": one_empty_self_count,
        "inner_empty_count": inner_empty_count,
        "side_empty_count": side_empty_count
    }


def get_shape_fast(board, x, y, dx, dy, role, size):
    # Optimization: Check immediate neighbours first (simplifed from JS version)
    nx1, ny1 = x + dx, y + dy
    nx2, ny2 = x - dx, y - dy
    valid1 = 0 <= nx1 < size and 0 <= ny1 < size
    valid2 = 0 <= nx2 < size and 0 <= ny2 < size

    # Quick check if surrounding are empty (less likely to form high shapes)
    # Note: JS version checks 2 steps away, this is slightly simpler
    if (not valid1 or board[nx1 + 1][ny1 + 1] == 0) and \
            (not valid2 or board[nx2 + 1][ny2 + 1] == 0):
        # Check further only if needed (more complex checks omitted for brevity here)
        pass  # Continue to full check, might be needed for isolated TWOs etc.

    # --- Main Shape Logic ---
    shape = shapes["NONE"]

    # Count pieces and patterns in both directions (-dx, -dy) and (dx, dy)
    left = count_shape(board, x, y, -dx, -dy, role, size)
    right = count_shape(board, x, y, dx, dy, role, size)

    # Combine counts (including the piece at x, y itself)
    self_count = left["self_count"] + right["self_count"] + 1
    total_length = left["total_length"] + right["total_length"] + 1
    no_empty_self_count = left["no_empty_self_count"] + right["no_empty_self_count"] + 1

    # Calculate OneEmptySelfCount carefully combining left and right parts
    # Max of (left side allowing one gap + right side contiguous) or (left contiguous + right side allowing one gap)
    one_empty_self_count = max(left["one_empty_self_count"] + right["no_empty_self_count"],
                               left["no_empty_self_count"] + right["one_empty_self_count"]) + 1

    left_empty = left["side_empty_count"]
    right_empty = right["side_empty_count"]

    # If the total potential line length is less than 5, cannot form significant shapes
    if total_length < 5:
        return shape, self_count  # Return NONE shape

    # --- Determine Shape based on counts ---

    # FIVE (win condition)
    if no_empty_self_count >= 5:
        # This check is slightly simplified; original JS might distinguish FIVE and BLOCK_FIVE more subtly
        # For game logic, >= 5 is usually a win. Score evaluation might differ.
        return shapes["FIVE"], self_count

    # FOUR
    if no_empty_self_count == 4:
        # Check if open on both ends (Live Four)
        if left_empty >= 1 and right_empty >= 1:
            return shapes["FOUR"], self_count
        # Check if blocked on one end (Blocked Four)
        elif left_empty >= 1 or right_empty >= 1:
            return shapes["BLOCK_FOUR"], self_count

    # Check for FOUR formed with one internal gap (e.g., X_XXX or XXX_X)
    if one_empty_self_count == 4:
        # Requires space on both sides of the gap pattern to be potentially live
        # Simplified: Treat patterns like X_XXX or XX_XX as Blocked Four for scoring
        if left_empty >= 1 and right_empty >= 1:  # Needs space around the gapped pattern
            return shapes["BLOCK_FOUR"], self_count  # Could argue FOUR if truly open, but B4 is safer eval

    # THREE
    if no_empty_self_count == 3:
        # Live Three: Needs enough space on both sides (e.g., _XXX_ _)
        # Requires at least one space on each side, and total spaces >= 2?
        is_left_open = left_empty >= 1 or (
                    left["inner_empty_count"] > 0 and left["side_empty_count"] >= 1)  # Space or gap+space
        is_right_open = right_empty >= 1 or (right["inner_empty_count"] > 0 and right["side_empty_count"] >= 1)

        if is_left_open and is_right_open and total_length >= 6:  # Needs room to become FOUR
            # Additional check: JS logic '001110' needs 2 spaces on one side.
            # e.g., left_empty >= 2 and right_empty >= 1 or vice-versa
            if (left_empty >= 2 and right_empty >= 1) or (left_empty >= 1 and right_empty >= 2):
                return shapes["THREE"], self_count
            else:
                return shapes["BLOCK_THREE"], self_count  # Not enough space for live 3
        # Blocked Three: Open on only one side
        elif is_left_open or is_right_open:
            return shapes["BLOCK_THREE"], self_count

    # Check for THREE formed with one internal gap (e.g., X_XX_ or _XX_X_)
    if one_empty_self_count == 3:
        # Live Three (Gapped): Needs space on both ends (e.g., _X_XX_)
        if left_empty >= 1 and right_empty >= 1 and total_length >= 6:  # Needs space around the gap pattern
            return shapes["THREE"], self_count
        # Blocked Three (Gapped)
        elif left_empty >= 1 or right_empty >= 1:
            return shapes["BLOCK_THREE"], self_count

    # TWO
    # Live Two (e.g., __XX__) - Needs significant space
    if no_empty_self_count == 2:
        is_left_open = left_empty >= 1
        is_right_open = right_empty >= 1
        if is_left_open and is_right_open and total_length >= 6:  # Needs room to grow
            # More specific check: e.g., _ _XX_ _ needs >= 2 spaces on each side?
            if (left_empty >= 2 and right_empty >= 2):  # Example: 001100
                return shapes["TWO"], self_count
            # Simpler TWO check (like JS 011000 or 000110)
            elif (left_empty >= 1 and right_empty >= 3) or (left_empty >= 3 and right_empty >= 1):
                return shapes["TWO"], self_count
            else:  # Potential Blocked Two
                return shapes["BLOCK_TWO"], self_count
        elif is_left_open or is_right_open:
            return shapes["BLOCK_TWO"], self_count

    # Live Two (Gapped, e.g., _X_X_)
    if one_empty_self_count == 2:
        is_left_open = left_empty >= 1
        is_right_open = right_empty >= 1
        if is_left_open and is_right_open and total_length >= 6:
            return shapes["TWO"], self_count
        elif is_left_open or is_right_open:
            return shapes["BLOCK_TWO"], self_count

    # ONE (less critical for eval, but for completeness)
    if no_empty_self_count == 1:
        is_left_open = left_empty >= 1
        is_right_open = right_empty >= 1
        if is_left_open and is_right_open and total_length >= 6:  # Needs space
            return shapes["ONE"], self_count
        elif is_left_open or is_right_open:
            return shapes["BLOCK_ONE"], self_count

    return shape, self_count  # Default is NONE


def get_all_shapes_of_point(shape_cache, x, y, role=None):
    roles_to_check = [role] if role is not None else [1, -1]
    result = []
    for r in roles_to_check:
        if r in shape_cache:
            for d in range(4):  # 4 directions
                if x < len(shape_cache[r][d]) and y < len(shape_cache[r][d][x]):  # Bounds check
                    shape = shape_cache[r][d][x][y]
                    if shape > shapes["NONE"]:
                        result.append(shape)
    return result


# --- Positions (positions.py equivalent) ---
def position_to_coordinate(position, size):
    return position // size, position % size


def coordinate_to_position(x, y, size):
    return x * size + y


# --- Cache (cache.py equivalent) ---
class Cache:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.cache_dict = {}
        self.cache_queue = deque()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if not config["enableCache"]:
            return None
        if key in self.cache_dict:
            self.hits += 1
            # Optional: Move accessed key to end of queue if using LRU
            # self.cache_queue.remove(key)
            # self.cache_queue.append(key)
            return self.cache_dict[key]
        else:
            self.misses += 1
            return None

    def put(self, key, value):
        if not config["enableCache"]:
            return
        if key not in self.cache_dict:
            if len(self.cache_queue) >= self.capacity:
                oldest_key = self.cache_queue.popleft()  # FIFO
                if oldest_key in self.cache_dict:
                    del self.cache_dict[oldest_key]
            self.cache_queue.append(key)
        # else: # Optional: If LRU, move existing key to end
        # self.cache_queue.remove(key)
        # self.cache_queue.append(key)

        self.cache_dict[key] = value

    def has(self, key):
        if not config["enableCache"]:
            return False
        return key in self.cache_dict


# --- Evaluation (eval.py equivalent) ---

# Score constants
FIVE_SCORE = 10000000
FOUR_SCORE = 100000
THREE_SCORE = 10000  # Increased from JS THREE=1000 for more impact
TWO_SCORE = 100
ONE_SCORE = 10

BLOCK_FOUR_SCORE = 10000  # Reduced from JS 1500? No, JS was 1500. Let's make it significant but less than live four
BLOCK_THREE_SCORE = 150
BLOCK_TWO_SCORE = 15
BLOCK_ONE_SCORE = 1

# Scores for combined shapes (derived)
FOUR_FOUR_SCORE = FOUR_SCORE * 2  # Example: prioritize double threats
FOUR_THREE_SCORE = FOUR_SCORE  # Very strong threat
THREE_THREE_SCORE = THREE_SCORE * 5  # JS had /2, making it stronger here relative to BLOCK_FOUR

TWO_TWO_SCORE = TWO_SCORE * 2  # Double live two is decent


# Maps a detected shape at an EMPTY square to the potential score gain IF that player moves there
def get_potential_score(shape):
    if shape == shapes["FIVE"]: return FOUR_SCORE  # Moving here creates FOUR threat (or wins if FIVE)
    if shape == shapes["FOUR"]: return THREE_SCORE  # Moving here creates THREE threat
    if shape == shapes["BLOCK_FOUR"]: return BLOCK_THREE_SCORE  # Moving here creates Blocked THREE
    if shape == shapes["THREE"]: return TWO_SCORE  # Moving here creates TWO threat
    if shape == shapes["BLOCK_THREE"]: return BLOCK_TWO_SCORE  # Moving here creates Blocked TWO
    if shape == shapes["TWO"]: return ONE_SCORE
    if shape == shapes["BLOCK_TWO"]: return BLOCK_ONE_SCORE
    if shape == shapes["ONE"]: return 0  # Minimal value

    # Combined shapes (more complex - evaluate based on components)
    # These scores reflect the value of *creating* this combined shape by placing one stone
    if shape == shapes["FOUR_FOUR"]: return FOUR_THREE_SCORE  # Making a move creates 4-3 threat usually
    if shape == shapes["FOUR_THREE"]: return FOUR_THREE_SCORE  # High threat
    if shape == shapes["THREE_THREE"]: return THREE_THREE_SCORE  # Creates a double three situation
    if shape == shapes["TWO_TWO"]: return TWO_TWO_SCORE  # Creates double two

    return 0


class Evaluate:
    def __init__(self, size=15):
        self.size = size
        # Board: 0 = empty, 1 = black, -1 = white, 2 = border
        self.board = [[2] * (size + 2)] + \
                     [[2] + [0] * size + [2] for _ in range(size)] + \
                     [[2] * (size + 2)]
        # Scores for empty points: potential score if black/white moves there
        self.black_scores = [[0] * size for _ in range(size)]
        self.white_scores = [[0] * size for _ in range(size)]
        self.history = []  # list of (x, y, role) tuples
        self.has_won = None  # Stores winning role (1 or -1) or 0 for draw, None if ongoing
        self.winning_line = []  # Stores the coordinates of the winning line

        # Cache: [role][direction][x][y] = shape
        # Roles: 1 (black), -1 (white)
        # Directions: 0:-, 1:|, 2:\, 3:/
        self.shape_cache = {
            1: [[([shapes["NONE"]] * size) for _ in range(size)] for _ in range(4)],
            -1: [[([shapes["NONE"]] * size) for _ in range(size)] for _ in range(4)],
        }
        self.all_directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # H, V, D\, D/

    def _direction_to_index(self, dx, dy):
        if dx == 0: return 0  # Vertical | (swapped index vs JS?) -> JS: 0=H, 1=V, 2=\, 3=/; Let's stick to JS:
        if dx == 0 and dy == 1: return 0  # Horizontal -
        if dx == 1 and dy == 0: return 1  # Vertical |
        if dx == 1 and dy == 1: return 2  # Diagonal \
        if dx == 1 and dy == -1: return 3  # Diagonal /
        # Handle negative directions too if needed, or ensure dx,dy are always positive representation
        if dx == 0 and dy == -1: return 0  # Horizontal -
        if dx == -1 and dy == 0: return 1  # Vertical |
        if dx == -1 and dy == -1: return 2  # Diagonal \
        if dx == -1 and dy == 1: return 3  # Diagonal /
        return -1  # Should not happen with self.all_directions

    def move(self, x, y, role):
        if not (0 <= x < self.size and 0 <= y < self.size and self.board[x + 1][y + 1] == 0):
            print(f"Warning: Invalid move attempted at ({x}, {y})")
            return False  # Invalid move

        self.board[x + 1][y + 1] = role
        self.history.append((x, y, role))

        # Check for win immediately after move
        if self.check_win(x, y, role):
            self.has_won = role
        elif len(self.history) == self.size * self.size:
            self.has_won = 0  # Draw

        # Update scores around the move
        self._update_scores_around(x, y)
        return True

    def undo(self):
        if not self.history:
            return False
        x, y, role = self.history.pop()
        self.board[x + 1][y + 1] = 0
        self.has_won = None  # Game is no longer won/drawn after undo
        self.winning_line = []

        # Update scores around the undone move
        self._update_scores_around(x, y)
        return True

    def check_win(self, x, y, role):
        """Checks if the last move at (x, y) by 'role' resulted in a win."""
        for dx, dy in self.all_directions:
            count = 1  # Count the piece just placed
            line_coords = [(x, y)]
            # Check positive direction
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx + 1][ny + 1] == role:
                    count += 1
                    line_coords.append((nx, ny))
                else:
                    break
            # Check negative direction
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx + 1][ny + 1] == role:
                    count += 1
                    line_coords.append((nx, ny))
                else:
                    break

            if count >= 5:
                self.winning_line = line_coords  # Store the winning line
                return True
        return False

    def is_game_over(self):
        """Returns True if game is won or drawn, False otherwise."""
        return self.has_won is not None

    def get_winner(self):
        """Returns winning role (1 or -1), 0 for draw, or None if ongoing."""
        return self.has_won

    def _update_scores_around(self, x, y):
        """Updates shape cache and potential scores for empty points around (x, y)."""
        # Clear the score/shape at the moved point itself (it's no longer empty)
        if 0 <= x < self.size and 0 <= y < self.size:
            self.black_scores[x][y] = 0
            self.white_scores[x][y] = 0
            for role in [1, -1]:
                for d in range(4):
                    self.shape_cache[role][d][x][y] = shapes["NONE"]

        # Iterate through directions and steps around the central point (x, y)
        for dx, dy in self.all_directions:
            for sign in [1, -1]:  # Positive and negative directions
                for step in range(1, 6):  # Check up to 5 steps away along the line
                    nx, ny = x + sign * step * dx, y + sign * step * dy

                    # Check bounds
                    if not (0 <= nx < self.size and 0 <= ny < self.size):
                        break  # Stop searching in this direction if out of bounds

                    # If we hit a non-empty square further out, its own score doesn't change,
                    # but empty squares *between* (x,y) and this square might need updating.
                    # However, the main impact is on the empty squares.

                    # Only update scores for currently EMPTY squares
                    if self.board[nx + 1][ny + 1] == 0:
                        # Update score for both black and white potential moves at this empty square (nx, ny)
                        self._update_single_point_score(nx, ny, 1)  # Update potential black score
                        self._update_single_point_score(nx, ny, -1)  # Update potential white score

                    # Optimization: If we hit a piece (of either color),
                    # squares further away in this exact line/direction are less likely
                    # to be immediately affected in terms of high-value shapes by the original move at (x,y).
                    # The JS code seems to continue, let's stick to that for now.
                    # If the point hit is not empty, we might still need to update points *beyond* it
                    # if the line continues. But for direct impact, updating closer empty cells is key.

                    # JS logic seems to stop at boundary or opponent piece only when checking *from* the empty cell.
                    # When updating *around* a placed piece, it iterates 5 steps regardless. Let's refine:

                    # We only need to re-evaluate empty points (nx, ny)
                    if self.board[nx + 1][ny + 1] != 0:
                        continue  # Skip non-empty points for score update

    def _update_single_point_score(self, x, y, role):
        """
        Calculates the potential score IF 'role' were to move at EMPTY square (x, y).
        Updates self.black_scores or self.white_scores[x][y].
        Also updates the shape_cache for this point and role.
        """
        if not (0 <= x < self.size and 0 <= y < self.size) or self.board[x + 1][y + 1] != 0:
            return 0  # Only score empty, valid points

        current_total_score = 0

        # Store shapes found in each direction for combination checks
        found_shapes = {0: shapes["NONE"], 1: shapes["NONE"], 2: shapes["NONE"], 3: shapes["NONE"]}

        # Simulate placing the piece to evaluate shapes formed
        self.board[x + 1][y + 1] = role

        shape_counts = {s: 0 for s in shapes.values()}

        for d_idx, (dx, dy) in enumerate(self.all_directions):
            # Get the shape formed in this direction by placing 'role' at (x, y)
            shape, _ = get_shape_fast(self.board, x, y, dx, dy, role, self.size)

            # Store the basic shape found for this direction in the cache
            self.shape_cache[role][d_idx][x][y] = shape
            found_shapes[d_idx] = shape

            if shape > shapes["NONE"]:
                shape_counts[shape] += 1  # Count occurrences of basic shapes

        # --- Calculate Combined Shapes ---
        # Check for combinations based on counts of basic shapes
        final_shape_for_score = shapes["NONE"]

        # Check for FIVE first (instant win potential) - should be handled by check_win mostly
        if shape_counts[shapes["FIVE"]] > 0 or shape_counts[shapes["BLOCK_FIVE"]] > 0:
            final_shape_for_score = shapes["FIVE"]  # Treat BlockFive as Five for eval here
        elif shape_counts[shapes["FOUR"]] >= 1:  # Live Four is highest priority after Five
            final_shape_for_score = shapes["FOUR"]
        elif shape_counts[shapes["BLOCK_FOUR"]] >= 2:  # Double Blocked Four -> Four Four
            final_shape_for_score = shapes["FOUR_FOUR"]
        elif shape_counts[shapes["BLOCK_FOUR"]] == 1 and shape_counts[
            shapes["THREE"]] >= 1:  # Blocked Four + Live Three -> Four Three
            final_shape_for_score = shapes["FOUR_THREE"]
        elif shape_counts[shapes["BLOCK_FOUR"]] >= 1:  # Single Blocked Four
            final_shape_for_score = shapes["BLOCK_FOUR"]
        elif shape_counts[shapes["THREE"]] >= 2:  # Double Live Three -> Three Three
            final_shape_for_score = shapes["THREE_THREE"]
        elif shape_counts[shapes["THREE"]] == 1:  # Single Live Three
            final_shape_for_score = shapes["THREE"]
        elif shape_counts[shapes["BLOCK_THREE"]] >= 1:  # Blocked Three (lower priority)
            final_shape_for_score = shapes["BLOCK_THREE"]  # Could check combinations with TWOs here too
        elif shape_counts[shapes["TWO"]] >= 2:  # Double Live Two -> Two Two
            final_shape_for_score = shapes["TWO_TWO"]
        elif shape_counts[shapes["TWO"]] == 1:  # Single Live Two
            final_shape_for_score = shapes["TWO"]
        elif shape_counts[shapes["BLOCK_TWO"]] >= 1:  # Blocked Two
            final_shape_for_score = shapes["BLOCK_TWO"]
        elif shape_counts[shapes["ONE"]] >= 1:  # Live One
            final_shape_for_score = shapes["ONE"]
        elif shape_counts[shapes["BLOCK_ONE"]]:  # Blocked One
            final_shape_for_score = shapes["BLOCK_ONE"]

        # Get the score contribution from the best single or combined shape identified
        current_total_score = get_potential_score(final_shape_for_score)

        # Undo the simulation
        self.board[x + 1][y + 1] = 0

        # Store the calculated potential score
        if role == 1:
            self.black_scores[x][y] = current_total_score
        else:
            self.white_scores[x][y] = current_total_score

        return current_total_score

    def evaluate_board(self, role):
        """Calculates the overall board score for the current 'role'."""
        if self.has_won == role: return FIVE_SCORE  # Maximize score if current player won
        if self.has_won == -role: return -FIVE_SCORE  # Minimize score if opponent won
        if self.has_won == 0: return 0  # Draw

        black_total_score = 0
        white_total_score = 0

        for r in range(self.size):
            for c in range(self.size):
                # Summing potential scores from empty squares
                black_total_score += self.black_scores[r][c]
                white_total_score += self.white_scores[r][c]

                # Optional: Add small score for pieces already on board?
                # if self.board[r+1][c+1] == 1: black_total_score += 1
                # if self.board[r+1][c+1] == -1: white_total_score += 1

        # The evaluation is relative to the 'role' whose turn it is in the minimax search
        if role == 1:  # Black's turn to evaluate
            return black_total_score - white_total_score * 1.0  # Opponent score weight (1.0 = equal)
        else:  # White's turn to evaluate
            return white_total_score - black_total_score * 1.0

    def get_valuable_moves(self, role, depth, only_three=False, only_four=False):
        """
        Generates a list of promising moves for the given role.
        Simplified version of the JS getPoints/getMoves logic.
        Focuses on moves that create significant shapes.
        Returns list of tuples: (score, x, y)
        """
        moves = []
        five_moves = []
        four_moves = []
        block_four_moves = []
        three_three_moves = []
        four_three_moves = []
        three_moves = []
        block_three_moves = []
        two_two_moves = []
        two_moves = []
        other_moves = []

        opponent = -role

        # Collect potential scores for all empty squares
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r + 1][c + 1] == 0:  # If square is empty
                    # Score if 'role' moves here
                    my_score = self.black_scores[r][c] if role == 1 else self.white_scores[r][c]
                    # Score if 'opponent' moves here (defensive value)
                    opponent_score = self.white_scores[r][c] if role == 1 else self.black_scores[r][c]

                    # Total score for considering this move: prioritize high offense and high defense
                    # Give slightly more weight to own potential score
                    total_score = my_score * 1.1 + opponent_score

                    if total_score > 0:
                        # Categorize based on the shape 'role' would create (using cached shapes)
                        # This requires checking the shape_cache *after* _update_single_point_score has run

                        # Re-evaluate the shape 'role' creates NOW at (r,c) to categorize
                        # This is slightly redundant but ensures categorization is correct
                        self.board[r + 1][c + 1] = role  # Simulate

                        best_shape_created = shapes["NONE"]
                        shape_counts = {s: 0 for s in shapes.values()}
                        for d_idx, (dx, dy) in enumerate(self.all_directions):
                            shape, _ = get_shape_fast(self.board, r, c, dx, dy, role, self.size)
                            if shape > shapes["NONE"]: shape_counts[shape] += 1

                        # Determine best combined shape (same logic as in _update_single_point_score)
                        if shape_counts[shapes["FIVE"]] > 0 or shape_counts[shapes["BLOCK_FIVE"]] > 0:
                            best_shape_created = shapes["FIVE"]
                        elif shape_counts[shapes["FOUR"]] >= 1:
                            best_shape_created = shapes["FOUR"]
                        elif shape_counts[shapes["BLOCK_FOUR"]] >= 2:
                            best_shape_created = shapes["FOUR_FOUR"]
                        elif shape_counts[shapes["BLOCK_FOUR"]] == 1 and shape_counts[shapes["THREE"]] >= 1:
                            best_shape_created = shapes["FOUR_THREE"]
                        elif shape_counts[shapes["BLOCK_FOUR"]] >= 1:
                            best_shape_created = shapes["BLOCK_FOUR"]
                        elif shape_counts[shapes["THREE"]] >= 2:
                            best_shape_created = shapes["THREE_THREE"]
                        elif shape_counts[shapes["THREE"]] == 1:
                            best_shape_created = shapes["THREE"]
                        elif shape_counts[shapes["BLOCK_THREE"]] >= 1:
                            best_shape_created = shapes["BLOCK_THREE"]
                        elif shape_counts[shapes["TWO"]] >= 2:
                            best_shape_created = shapes["TWO_TWO"]
                        elif shape_counts[shapes["TWO"]] == 1:
                            best_shape_created = shapes["TWO"]
                        else:
                            best_shape_created = shapes["NONE"]  # Or TWO/ONE if needed

                        self.board[r + 1][c + 1] = 0  # Undo simulation

                        move_tuple = (total_score, r, c)

                        # --- Categorization based on shape created by 'role' ---
                        # 1. Winning Moves (creates FIVE) - Highest Priority
                        if best_shape_created == shapes["FIVE"]:
                            five_moves.append(move_tuple)
                            continue  # Don't add to lower categories

                        # 2. Blocking Opponent's FIVE - Also Highest Priority
                        # Check if opponent moving here would create FIVE
                        self.board[r + 1][c + 1] = opponent  # Simulate opponent
                        opp_creates_five = False
                        for d_idx, (dx, dy) in enumerate(self.all_directions):
                            shape, _ = get_shape_fast(self.board, r, c, dx, dy, opponent, self.size)
                            if shape == shapes["FIVE"]:
                                opp_creates_five = True
                                break
                        self.board[r + 1][c + 1] = 0  # Undo simulation
                        if opp_creates_five:
                            # Prioritize blocking win even higher than making own FOUR
                            five_moves.append((total_score + FIVE_SCORE, r, c))  # Add huge bonus
                            continue

                            # 3. Creating Live FOUR
                        if best_shape_created == shapes["FOUR"]:
                            four_moves.append(move_tuple)
                            continue

                        # 4. Creating Blocked FOUR or complex threats (4-4, 4-3) that aren't FOUR
                        if best_shape_created in [shapes["BLOCK_FOUR"], shapes["FOUR_FOUR"], shapes["FOUR_THREE"]]:
                            block_four_moves.append(move_tuple)
                            continue

                        # 5. Creating Double Three
                        if best_shape_created == shapes["THREE_THREE"]:
                            three_three_moves.append(move_tuple)
                            continue

                        # 6. Creating Live Three
                        if best_shape_created == shapes["THREE"]:
                            three_moves.append(move_tuple)
                            continue

                        # Filter based on only_three/only_four flags (simplified from JS VCT/VCF logic)
                        if only_four: continue  # If only_four, ignore lower moves now
                        if only_three and best_shape_created < shapes["THREE"]: continue  # If only_three, ignore lower

                        # 7. Creating Blocked Three
                        if best_shape_created == shapes["BLOCK_THREE"]:
                            block_three_moves.append(move_tuple)
                            continue

                        # 8. Creating Double Two
                        if best_shape_created == shapes["TWO_TWO"]:
                            two_two_moves.append(move_tuple)
                            continue

                        # 9. Creating Live Two
                        if best_shape_created == shapes["TWO"]:
                            two_moves.append(move_tuple)
                            continue

                        # 10. Other moves with some score (Blocked Two, One, etc.)
                        other_moves.append(move_tuple)

        # Sort each category by score (descending)
        five_moves.sort(key=lambda item: item[0], reverse=True)
        four_moves.sort(key=lambda item: item[0], reverse=True)
        block_four_moves.sort(key=lambda item: item[0], reverse=True)
        three_three_moves.sort(key=lambda item: item[0], reverse=True)
        four_three_moves.sort(key=lambda item: item[0], reverse=True)  # Added category
        three_moves.sort(key=lambda item: item[0], reverse=True)
        block_three_moves.sort(key=lambda item: item[0], reverse=True)
        two_two_moves.sort(key=lambda item: item[0], reverse=True)
        two_moves.sort(key=lambda item: item[0], reverse=True)
        other_moves.sort(key=lambda item: item[0], reverse=True)

        # Combine lists in order of priority
        # If winning moves exist, only return those
        if five_moves:
            return [(x, y) for score, x, y in five_moves[:config["pointsLimit"]]]

        # Combine offensive and defensive FOUR moves (blocking opponent FOUR is crucial)
        # Need to check opponent's potential FOURs similar to checking their FIVEs above

        # Simplified approach: Combine categories

        # Priority Order: FOUR, BLOCK_FOUR/4-3/4-4, THREE_THREE, THREE, BLOCK_THREE, TWO_TWO, TWO, OTHER
        ordered_moves = four_moves + block_four_moves + four_three_moves + three_three_moves + three_moves \
                        + block_three_moves + two_two_moves + two_moves + other_moves

        # Apply pointsLimit constraint (mainly affects lower priority moves)
        return [(x, y) for score, x, y in ordered_moves[:config["pointsLimit"]]]

    def get_empty_points(self):
        """Returns all empty points - fallback if valuable moves are empty"""
        empty = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r + 1][c + 1] == 0:
                    empty.append((r, c))
        # Optional: shuffle to avoid predictable play in empty situations
        random.shuffle(empty)
        return empty

    def board_hash(self):
        """Generates a hashable representation of the board state for caching."""
        # Using a tuple of tuples for the relevant board part
        return tuple(tuple(row[1:-1]) for row in self.board[1:-1])


# --- Minimax Search (minimax.py equivalent) ---
cache = Cache()  # Global cache instance for the search
MAX_SCORE = FIVE_SCORE * 2  # Use a score higher than winning score
MIN_SCORE = -MAX_SCORE


# --- Negamax Implementation with Alpha-Beta Pruning ---
def minimax_search(board_state, role, depth, alpha, beta, current_depth=0):
    """
    Negamax search function.
    board_state: The Evaluate object representing the current game state.
    role: The player whose turn it is (1 for black, -1 for white).
    depth: The maximum search depth remaining.
    alpha: Alpha value for pruning.
    beta: Beta value for pruning.
    current_depth: The depth reached so far from the root.
    Returns: (best_score, best_move_tuple (x, y))
    """
    global cache

    # --- Caching ---
    board_hash_key = (board_state.board_hash(), role, depth)  # Include role and depth in key
    cached_result = cache.get(board_hash_key)
    if cached_result:
        # Ensure cached score is adjusted for current player's perspective if needed (Negamax handles this inherently)
        # We need to store the score relative to the player *whose turn it was* when cached.
        # And return it directly.
        # print(f"Cache hit at depth {current_depth}, remaining {depth}")
        return cached_result["score"], cached_result["move"]  # Return cached score and move

    # --- Terminal State Check ---
    if board_state.is_game_over():
        winner = board_state.get_winner()
        if winner == role:
            score = FIVE_SCORE  # Current player won
        elif winner == -role:
            score = -FIVE_SCORE  # Opponent won
        else:
            score = 0  # Draw
        # Adjust score slightly by depth: prefer quicker wins, later losses
        score -= current_depth  # Penalize score the deeper the win/loss is
        return score, None  # No move to make from a terminal state

    if depth <= 0:
        # Evaluate board from the perspective of the current 'role'
        score = board_state.evaluate_board(role)
        return score, None  # No move, just evaluation score

    # --- Move Generation ---
    # Get valuable moves first
    possible_moves = board_state.get_valuable_moves(role, current_depth)

    # If no valuable moves, consider all empty squares (or a random subset)
    if not possible_moves:
        possible_moves = board_state.get_empty_points()
        # Limit the number of empty points to check if there are too many
        if len(possible_moves) > config["pointsLimit"] * 2:
            possible_moves = random.sample(possible_moves, config["pointsLimit"] * 2)

    if not possible_moves:  # No moves possible at all (should only happen on full board draw already checked)
        return 0, None  # Draw score

    # --- Search ---
    best_score = MIN_SCORE
    best_move = None

    # Start with the first move if available, otherwise random/center
    if not best_move and possible_moves:
        best_move = possible_moves[0]

    for move in possible_moves:
        x, y = move

        # Make the move
        if not board_state.move(x, y, role):
            # print(f"Skipping invalid move {move} in search")
            continue  # Should not happen if move generation is correct

        # Recursive call for the opponent (-role), swapping alpha/beta and negating
        score, _ = minimax_search(board_state, -role, depth - 1, -beta, -alpha, current_depth + 1)
        score = -score  # Negate score returned by opponent

        # Undo the move
        board_state.undo()

        # --- Update Best Score and Alpha/Beta ---
        if score > best_score:
            best_score = score
            best_move = move  # Update best move found so far

        alpha = max(alpha, best_score)  # Update alpha (lower bound)

        # --- Alpha-Beta Pruning ---
        if alpha >= beta:
            # print(f"Pruning at depth {current_depth} (alpha={alpha}, beta={beta})")
            break  # Prune remaining moves at this node

    # --- Cache the result before returning ---
    if config["enableCache"]:
        # Cache score relative to the player (role) whose turn it was at this node
        cache.put(board_hash_key, {"score": best_score, "move": best_move})

    return best_score, best_move


def find_best_move(board_state, role, search_depth):
    """Top-level function to initiate the search."""
    global cache
    cache.hits = 0
    cache.misses = 0

    start_time = time.time()

    # If board is empty, play center or near center
    if not board_state.history:
        cx, cy = board_state.size // 2, board_state.size // 2
        # Add slight randomness near center for opening variety
        rx = random.randint(-1, 1)
        ry = random.randint(-1, 1)
        return (cx + rx, cy + ry)

    best_score, best_move = minimax_search(board_state, role, search_depth, MIN_SCORE, MAX_SCORE)

    end_time = time.time()
    print(f"AI ({'Black' if role == 1 else 'White'}) searched depth {search_depth}.")
    print(f"Best score: {best_score}. Best move: {best_move}.")
    print(f"Search time: {end_time - start_time:.2f} seconds.")
    if config["enableCache"]:
        total_access = cache.hits + cache.misses
        hit_rate = (cache.hits / total_access * 100) if total_access > 0 else 0
        print(f"Cache: Hits={cache.hits}, Misses={cache.misses}, Hit Rate={hit_rate:.1f}%")

    # Fallback if search somehow fails to find a move
    if best_move is None:
        print("Warning: AI search failed to find a move. Picking random.")
        available = board_state.get_empty_points()
        if available:
            best_move = random.choice(available)
        else:  # No moves possible (shouldn't happen if game ends correctly)
            return None

    return best_move


# --- GUI (using Tkinter) ---
class GomokuGUI:
    def __init__(self, master, size=15, search_depth=4):
        self.master = master
        self.size = size
        self.cell_width = 40  # Pixel size of each cell
        self.board_pixels = self.size * self.cell_width
        self.padding = 20  # Padding around the board
        self.search_depth = search_depth

        self.master.title("Gomoku AI (Minimax + AlphaBeta)")

        # Game State
        self.board = Evaluate(size=self.size)
        self.current_player = 1  # 1 for Black (Human), -1 for White (AI) - Black starts
        self.ai_player = -1  # AI plays White
        self.game_over = False
        self.ai_thinking = False

        # Canvas for board
        self.canvas = tk.Canvas(master,
                                width=self.board_pixels + 2 * self.padding,
                                height=self.board_pixels + 2 * self.padding,
                                bg='#D2B48C')  # Wood-like background
        self.canvas.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(master, text="Your turn (Black)", font=("Arial", 14))
        self.status_label.pack()

        # New Game Button
        self.new_game_button = tk.Button(master, text="New Game", command=self.reset_game)
        self.new_game_button.pack(pady=10)

        # Bind click event
        self.canvas.bind("<Button-1>", self.handle_click)

        self.draw_board()
        # No initial pieces drawn

    def draw_board(self):
        self.canvas.delete("all")  # Clear previous drawings
        grid_start = self.padding
        grid_end = self.board_pixels + self.padding

        # Draw grid lines
        for i in range(self.size):
            x = grid_start + i * self.cell_width
            # Vertical lines (use +0.5 for sharper lines potentially)
            self.canvas.create_line(x, grid_start, x, grid_end, fill="black")
            # Horizontal lines
            self.canvas.create_line(grid_start, x, grid_end, x, fill="black")

        # Draw thicker outer border lines
        self.canvas.create_rectangle(grid_start, grid_start, grid_end, grid_end, outline="black", width=2)

        # Optional: Draw star points (common on Gomoku boards)
        if self.size == 15:
            star_points = [(3, 3), (11, 3), (3, 11), (11, 11), (7, 7)]
            for r, c in star_points:
                self.draw_star_point(r, c)
        elif self.size == 19:  # Example for 19x19
            star_points = [(3, 3), (9, 3), (15, 3), (3, 9), (9, 9), (15, 9), (3, 15), (9, 15), (15, 15)]
            for r, c in star_points:
                self.draw_star_point(r, c)

    def draw_star_point(self, row, col):
        center_x = self.padding + col * self.cell_width
        center_y = self.padding + row * self.cell_width
        radius = 4
        self.canvas.create_oval(center_x - radius, center_y - radius,
                                center_x + radius, center_y + radius, fill="black")

    def draw_pieces(self):
        self.canvas.delete("piece")  # Remove only pieces, keep grid
        radius = self.cell_width // 2 - 3  # Piece radius

        for r in range(self.size):
            for c in range(self.size):
                player = self.board.board[r + 1][c + 1]
                if player != 0:
                    center_x = self.padding + c * self.cell_width
                    center_y = self.padding + r * self.cell_width

                    fill_color = "black" if player == 1 else "white"
                    outline_color = "gray" if player == -1 else "black"  # White pieces need outline

                    self.canvas.create_oval(center_x - radius, center_y - radius,
                                            center_x + radius, center_y + radius,
                                            fill=fill_color, outline=outline_color, tags="piece")

        # Highlight the last move
        if self.board.history:
            last_x, last_y, _ = self.board.history[-1]
            center_x = self.padding + last_y * self.cell_width
            center_y = self.padding + last_x * self.cell_width
            mark_radius = 3
            self.canvas.create_oval(center_x - mark_radius, center_y - mark_radius,
                                    center_x + mark_radius, center_y + mark_radius,
                                    fill="red", outline="red", tags="piece")  # Mark last move

        # Highlight winning line if game is over
        if self.game_over and self.board.winning_line:
            line_coords_pixels = []
            for r, c in self.board.winning_line:
                center_x = self.padding + c * self.cell_width
                center_y = self.padding + r * self.cell_width
                line_coords_pixels.extend([center_x, center_y])

            # Draw a thick line through the centers of winning pieces
            if len(line_coords_pixels) >= 4:  # Need at least 2 points for a line
                self.canvas.create_line(line_coords_pixels, fill="blue", width=5, tags="piece")

    def handle_click(self, event):
        if self.game_over or self.ai_thinking:
            return  # Ignore clicks if game ended or AI is busy

        # Convert pixel coordinates to board coordinates
        # Account for padding and cell width, find nearest intersection
        col = round((event.x - self.padding) / self.cell_width)
        row = round((event.y - self.padding) / self.cell_width)

        # print(f"Click at ({event.x}, {event.y}) -> Board ({row}, {col})")

        if 0 <= row < self.size and 0 <= col < self.size:
            if self.board.board[row + 1][col + 1] == 0:
                # Human player's turn
                if self.current_player != self.ai_player:
                    if self.board.move(row, col, self.current_player):
                        self.draw_pieces()  # Update display immediately

                        if self.check_game_over():
                            return  # Stop if human won or draw

                        # Switch to AI player
                        self.current_player = self.ai_player
                        self.status_label.config(text="AI's turn (White) thinking...")
                        self.master.update()  # Update GUI to show thinking status
                        self.trigger_ai_move()
            else:
                print("Cell already occupied.")
        else:
            print("Clicked outside board boundaries.")

    def trigger_ai_move(self):
        self.ai_thinking = True
        # Use after to allow GUI to update before starting potentially long AI calculation
        self.master.after(100, self.perform_ai_move)

    def perform_ai_move(self):
        # Find best move using minimax
        ai_move = find_best_move(self.board, self.ai_player, self.search_depth)

        self.ai_thinking = False

        if ai_move:
            moved = self.board.move(ai_move[0], ai_move[1], self.ai_player)
            if moved:
                self.draw_pieces()
                if self.check_game_over():
                    return

                # Switch back to Human player
                self.current_player = 1  # Assuming human is always black=1
                self.status_label.config(text="Your turn (Black)")
            else:
                print(f"Error: AI tried invalid move {ai_move}")
                self.status_label.config(text="AI Error! Your turn (Black)")
                self.current_player = 1
        else:
            # This case should ideally not happen if game over check is robust
            print("AI could not find a move (or game already ended?)")
            self.check_game_over()  # Double check game state

    def check_game_over(self):
        if self.board.is_game_over():
            self.game_over = True
            winner = self.board.get_winner()
            if winner == 1:
                message = "You win (Black)!"
            elif winner == -1:
                message = "AI wins (White)!"
            else:
                message = "It's a draw!"
            self.status_label.config(text=message)
            self.draw_pieces()  # Redraw to show winning line if any
            tk.messagebox.showinfo("Game Over", message)
            return True
        return False

    def reset_game(self):
        self.board = Evaluate(size=self.size)  # Create a new board instance
        self.current_player = 1  # Black starts
        self.game_over = False
        self.ai_thinking = False
        self.status_label.config(text="Your turn (Black)")
        self.draw_board()  # Redraw empty board
        self.draw_pieces()  # Clear pieces


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root, size=config["board_size"], search_depth=config["depth"])
    root.mainloop()
