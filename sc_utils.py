
NUMBER_SET = [str(num) for num in range(0, 10)]

def _is_float(s):
  try:
    float(s)
    return True
  except:
    return False

FINAL_ANS = 'answer is '
def clean_ans(ans):
  index = ans.find('.')
  if index >= 0:
    end_index = index + 1
    while end_index < len(ans) and ans[end_index] in NUMBER_SET:
      end_index += 1
    ans = ans[:end_index]
  while ans and ans.endswith('.'):
    ans = ans[:-1]
  
  ans = ans.split('=')[-1].strip()
  for c in ['$', ',', '%', '€', '"']:
    ans = ans.replace(c, '')
  parts = ans.split(' ')
  for part in parts:
    if _is_float(part):
      return part
  
  ans = parts[0]  # default
  for part in parts:
    if not part.isalpha():  # take the 1st non-alpha token
      ans = part
      break
  while ans and ans[-1].isalpha():
    ans = ans[:-1]
  return ans.strip()

def get_ans(pred):
  text = pred.split('Q:')[0].split('[eot]')[0].replace('\n', '').strip()
  if text.rfind(FINAL_ANS) >= 0:
    pred_ans = text[text.rfind(FINAL_ANS) + len(FINAL_ANS):len(text)].strip()
    return clean_ans(pred_ans)
  else:
    return ''


from collections import Counter
def get_maj(ans_list):
  is_all_float = True
  float_list = []
  for ans in ans_list:
    if _is_float(ans):
      float_list.append(float(ans))
    else:
      is_all_float = False
      break
  if is_all_float:
    f = Counter(float_list)
    return f.most_common()[0][0]
  else:
    c = Counter(ans_list)
    return c.most_common()[0][0]

def get_str_ans(pred):
  text = pred.split('Q:')[0].split('[eot]')[0].replace('\n', '').strip()
  if text.rfind(FINAL_ANS) >= 0:
    pred_ans = text[text.rfind(FINAL_ANS) + len(FINAL_ANS):len(text)].strip()
    if pred_ans.endswith('.'):
      pred_ans = pred_ans[:-1]
    return pred_ans
  else:
    return ''