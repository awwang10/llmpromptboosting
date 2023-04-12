from datasets import load_dataset
from collections import defaultdict
import numpy as np
import os
import json


def remove_boxed(string: str):
    """Source: https://github.com/hendrycks/math

    Extract the text within a \\boxed{...} environment.

    Example:
    >>> remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str):
    """Source: https://github.com/hendrycks/math

    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def get_answer(solution):
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer

if __name__ == '__main__':
  np.random.seed(0)

  # Make Test Set

  problems = defaultdict(lambda: [])
  dataset = load_dataset('competition_math', split='test')


  for p in dataset:
    a = get_answer(p['solution'])
    try:
      assert '_' not in a
      assert '\n' not in p['problem']
      z = float(a)
      assert z.is_integer()
    except:
      continue
    p['answer'] = get_answer(p['solution'])
    problems[(p['level'],p['type'])].append(p)
    
  res = []

  for p in problems.values():
    assert len(p) >= 12
    res += list(np.random.choice(p, 12, replace=False))
    
  np.random.shuffle(res)

  if not os.path.exists(os.path.join('dataset','cmath420')):
    os.makedirs(os.path.join('dataset','cmath420'))
    
  with open(os.path.join('dataset','cmath420','cmath420.json'), 'w') as f:
    json.dump(res, f)
    
  # Make Prompt

  dataset = load_dataset('competition_math', split='train')

  candidates = []
  for p in dataset:
    if p['level'] == 'Level 1' and p['type'] == 'Prealgebra':
      a = get_answer(p['solution'])
      try:
        assert '_' not in a
        assert '\n' not in p['solution']
        assert '\n' not in p['problem']
        z = float(a)
        assert z.is_integer()
      except:
        continue
      p['answer'] = get_answer(p['solution'])
      candidates.append(p)
      
  prompt = np.random.choice(candidates, 8, replace=False)

  s = '\n\n'.join(f"Q: {p['problem']}\nA: {p['solution']} The answer is {p['answer']}." for p in prompt)

  if not os.path.exists('prompts'):
    os.makedirs('prompts')
    
  with open(os.path.join('prompts','cmath420.txt'), 'w') as f:
    f.write(s)