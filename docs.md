## Dataset Format
Each dataset record should contain the following fields:
- `question`
- `question_with_prompt`
- `answer`
- `choices`
There should be a scipt for each dataset to create a csv in scripts in correct folder.

## Completions File Structure
Each completion entry stores the token-matching metadata:
- `match_tokens`: List of matched token IDs. Use the fallback token when no match is found.
- `match_indices`: List of positions corresponding to the first match. Use the fallback index when no match exists.
- `match_flags`: List of booleans indicating whether each match was an exact match.

## Activation Cache Structure
The `activations` dictionary exposes cached model telemetry:
- `activations_cache`: Maps integer layer indices to NumPy arrays of activations.
- `y_correct`: Boolean list signaling whether each prediction was correct.
- `y_error_sm`: NumPy array of softmax errors per sample.
- `y_error_ce`: List of per-sample cross-entropy errors.
- `activations_cache_exact`: Mirror of `activations_cache` limited to exact matches.
- `y_correct_exact`: Boolean list for correctness restricted to exact matches.
- `y_error_sm_exact`: Softmax errors for exact matches.
- `y_error_ce_exact`: Cross-entropy errors for exact matches.

### Example
Accessing the activations for layer 0: