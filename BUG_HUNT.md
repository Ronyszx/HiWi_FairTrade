# Bug Hunt Notes

## Candidate Bug 1: Inconsistent return values from evaluate()

### Location
File: `FairTrade.py`  
Function/block: main optimization loop around the call to `evaluate()`  
Observed line: `objectives, bal_acc_, fairness_notion_ = evaluate(alpha)`

### Problem
The `evaluate()` function returns only one value: the `objectives` tensor.

However, the caller tries to unpack three values:

```python
objectives, bal_acc_, fairness_notion_ = evaluate(alpha)
```

This causes the baseline Adult experiment to crash with:

```text
ValueError: not enough values to unpack (expected 3, got 1)
```

### Why this matters

The training starts successfully and computes the first metrics, but the optimization loop cannot continue. This prevents reproducing the baseline FairTrade experiment without modifying the code.

### Proposed fix

The smallest safe fix is to assign only the returned `objectives` tensor:

```python
if round == 0:
    objectives = evaluate(alpha)
else:
    objectives = evaluate(updated_alpha, updated_lr)
```

This is sufficient because the code later reads fairness and balanced accuracy directly from the `objectives` tensor:

```python
fairness_notion_list.append(objectives[0,0].item())
bal_acc_list.append(objectives[0,1].item())
```

### Status

Implemented as a minimal runtime fix.
