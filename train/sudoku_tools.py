"""Hooks that need to be implemented for the Sudoku training pipeline.

The RL objective relies on three environment specific functions.  They are kept
in a dedicated module so that users can override them without having to touch
training code.  The default implementations raise :class:`NotImplementedError`
with detailed guidance which should make it straightforward to plug-in project
specific logic.
"""
from __future__ import annotations

import re
import numpy as np
from typing import Iterable, Sequence
DIGIT_BASE_ID = 15
SPACE_ID = 220
NEWLINE_ID = 198
MASK_ID = 126336
GRID = 6

DIGIT_TOKEN_MAP = {
    16: 1, 17: 2, 18: 3, 19: 4, 20: 5, 21: 6,
}

def encode_sudoku_prefill(text: str):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    ids = []
    for ch in text:
        if ch == '.':
            ids.append(MASK_ID)
        elif ch == ' ':
            ids.append(SPACE_ID)
        elif ch == '\n':
            ids.append(NEWLINE_ID)
        elif '0' <= ch <= '6':
            ids.append(DIGIT_BASE_ID + (ord(ch) - ord('0')))
        else:
            continue
    return ids


def pre_fill(prompt: str, tokenizer, max_gen_length: int):
    """Fill deterministic tokens before sampling starts.

    Parameters
    ----------
    prompt:
        Human readable description of the Sudoku instance.
    token_ids:
        Tokenised representation of ``prompt`` including mask tokens that mark
        Sudoku slots.

    Returns
    -------
    filled_ids: list[int]
        Sequence with the deterministic values inserted.
    filled_count: int
        Number of tokens that were filled.
    filled_indices: Iterable[int]
        Indices (relative to the first token after the prompt) that were filled
        by this routine.
    """
    m = [{"role": "user", "content": prompt}, ]
    all_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    all_prompt += """Sure, I'll help you solve the 6x6 Sudoku puzzle. Here is the completed Sudoku grid:\n```\n"""
    prompt_token_ids = tokenizer(all_prompt, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_token_ids)
    
    lines = prompt.strip().split('\n')
    puzzle_lines = []
    
    for line in lines:
        if re.match(r'^[0-6\.\s]+$', line.strip()):
            puzzle_lines.append(line.strip())
    
    puzzle_text = '\n'.join(puzzle_lines[:GRID])
    
    sudoku_token_ids = encode_sudoku_prefill(puzzle_text)
    sudoku_length = len(sudoku_token_ids)
    
    sudoku_map_range = (prompt_length, prompt_length + sudoku_length)
    
    filled_ids = list(prompt_token_ids)  
    filled_ids.extend(sudoku_token_ids)
    
    current_gen_length = len(filled_ids) - prompt_length  
    remaining_length = max_gen_length - current_gen_length
    
    if remaining_length > 0:
        filled_ids.extend([MASK_ID] * remaining_length)
    elif remaining_length < 0:
        filled_ids = filled_ids[:prompt_length + max_gen_length]
    
    filled_indices = []
    for i in range(len(sudoku_token_ids)):
        if sudoku_token_ids[i] != MASK_ID:
            filled_indices.append(i + prompt_length)
    
    return filled_ids, filled_indices, prompt_length, sudoku_map_range


def detect_definite(state_ids: Sequence[int]):
    """Identify positions that can be filled with certainty.

    The function must return an iterable that is understood by
    :func:`train.sudoku_rl_utils.normalise_definite_positions`.  Returning a list
    of ``(index, token_id)`` tuples is the most convenient option.
    """
    grid = np.zeros((GRID, GRID), dtype=int)
    mask_positions = []
    pos = 0
    
    for i, token_id in enumerate(state_ids):
        if token_id in DIGIT_TOKEN_MAP:
            row = pos // GRID
            col = pos % GRID
            if row < GRID and col < GRID:
                grid[row, col] = DIGIT_TOKEN_MAP[token_id]
            pos += 1
        elif token_id == MASK_ID:
            row = pos // GRID
            col = pos % GRID
            if row < GRID and col < GRID:
                mask_positions.append((row, col))
                grid[row, col] = 0
            pos += 1
        elif token_id == NEWLINE_ID:
            pass
        elif token_id == SPACE_ID:
            pass
    
    definite_positions = []
    
    for row, col in mask_positions:
        if grid[row, col] != 0:
            continue
            
        candidates = {1, 2, 3, 4, 5, 6}
        candidates -= set(grid[row, :])
        candidates -= set(grid[:, col])
        
        box_row = (row // 2) * 2
        box_col = (col // 3) * 3
        candidates -= set(grid[box_row:box_row+2, box_col:box_col+3].flatten())
        
        if len(candidates) == 1:
            definite_value = candidates.pop()
            token_index = find_tokenindex(state_ids, row, col)
            if token_index is not None:
                target_token_id = DIGIT_BASE_ID + definite_value
                definite_positions.append((token_index, target_token_id))
    
    return definite_positions

def find_tokenindex(token_ids: Sequence[int], target_row: int, target_col: int) -> int:
    
    pos = 0
    for i, token_id in enumerate(token_ids):
        if token_id in DIGIT_TOKEN_MAP or token_id == MASK_ID:
            row = pos // GRID
            col = pos % GRID
            if row == target_row and col == target_col:
                return i
            pos += 1
        elif token_id == NEWLINE_ID:
            pass
        elif token_id == SPACE_ID:
            pass
    return None

def judge_error(state_ids: Sequence[int]) -> bool:
    """Return ``True`` if the partially decoded Sudoku violates any constraint."""
    grid = np.zeros((GRID, GRID), dtype=int)
    pos = 0
    
    for token_id in state_ids:
        if token_id in DIGIT_TOKEN_MAP:
            row = pos // GRID
            col = pos % GRID
            if row < GRID and col < GRID:
                grid[row, col] = DIGIT_TOKEN_MAP[token_id]
            pos += 1
        elif token_id == MASK_ID:
            pos += 1
        elif token_id == NEWLINE_ID:
            pass
        elif token_id == SPACE_ID:
            pass
    # 检查行冲突
    for row in range(GRID):
        row_values = grid[row, :]
        non_zero = row_values[row_values != 0]
        if len(non_zero) != len(set(non_zero)):
            return True
    # 检查列冲突
    for col in range(GRID):
        col_values = grid[:, col]
        non_zero = col_values[col_values != 0]
        if len(non_zero) != len(set(non_zero)):
            return True
    # 检查宫格冲突
    for box_row in range(0, GRID, 2):
        for box_col in range(0, GRID, 3):
            box = grid[box_row:box_row+2, box_col:box_col+3]
            non_zero = box[box != 0]
            if len(non_zero) != len(set(non_zero)):
                return True
    
    return False

# 不知道是否需要，先放这里了
def format_6x6_sudoku_prompt(puzzle_text):
    
    prompt = """Please solve the following 6x6 sudoku puzzle.

Rules of 6x6 sudoku:
- The grid is 6x6.
- Each row must contain the digits 1-6 exactly once.
- Each column must contain the digits 1-6 exactly once.
- Each of the 6 subgrids (2x3 boxes) must also contain the digits 1-6 exactly once.
- The 6 subgrids are arranged as:
  - Box 1: rows 1-2, columns 1-3
  - Box 2: rows 1-2, columns 4-6  
  - Box 3: rows 3-4, columns 1-3
  - Box 4: rows 3-4, columns 4-6
  - Box 5: rows 5-6, columns 1-3
  - Box 6: rows 5-6, columns 4-6
- Empty cells are represented by a dot ".".

Here is the puzzle (6 lines, each with 6 entries separated by spaces):

""" + puzzle_text + """

Please output the completed sudoku as 36 numbers, row by row, separated by spaces."""
    
    return prompt