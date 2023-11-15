import numpy as np
import pandas as pd
import torch

class DataSet:
  def __init__(self, X, Y):
    self.input_ids = X['input_ids']
    self.attention_mask = X['attention_mask']
    self.token_type_ids = X['token_type_ids']
    self.labels = Y

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return {'input_ids': self.input_ids[index], 
            'attention_mask': self.attention_mask[index], 
            'token_type_ids': self.token_type_ids[index], 
            'labels': self.labels[index]}

upper_score_dic = {
  '2-3_1_5': {
      'score': 16,
      'A_Score': 3,
      'B_Score': 3,
      'C_Score': 2, 
      'D_Score': 4,
      'E_Score': 4
  },
  '2-3_2_2': {
      'score': 12,
      'A_Score': 3,
      'B_Score': 2,
      'C_Score': 2, 
      'D_Score': 3,
      'E_Score': 2
  },
  '2-3_2_4': {
      'score': 14,
      'A_Score': 2,
      'B_Score': 3,
      'C_Score': 3, 
      'D_Score': 4,
      'E_Score': 2
  },  
  '1-2_1_3': { 
      'score': 16,
      'A_Score': 2,
      'B_Score': 5,
      'C_Score': 3, 
      'D_Score': 6,
  },
  '1-2_2_4': { 
      'score': 12,
      'A_Score': 3,
      'B_Score': 2,
      'C_Score': 4, 
      'D_Score': 3,
  },
  '2-1_1_5': { 
      'score': 15,
      'A_Score': 2,
      'B_Score': 7,
      'C_Score': 6, 
  },
  '2-1_2_3': { 
      'score': 12,
      'A_Score': 3,
      'B_Score': 4,
      'C_Score': 3, 
      'D_Score': 2,
  },
  '2-2_1_4': { 
      'score': 15,
      'A_Score': 6,
      'B_Score': 3,
      'C_Score': 6, 
  },
  '2-2_2_3': { 
      'score': 14,
      'A_Score': 6,
      'B_Score': 6,
      'C_Score': 2, 
  }
}

def get_upper_score(q_id, s_id):
     return upper_score_dic[q_id][s_id]

def get_dataset(dataf, s_id, upper_s, inftype, tokenizer):
  text, score = [], []
  for data in dataf:
    text.append(data['mecab'].replace(' ', ''))
    score.append(data[s_id])
  encoding = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
  if inftype == 'reg' or inftype == 'mix':
    labels = torch.div(torch.tensor(score, dtype=torch.float32), upper_s)
  elif inftype == 'class':
    labels = torch.tensor(score, dtype=torch.long)
  else:
    raise ValueError("{} is not a valid value for scoretype".format(inftype))
  return DataSet(encoding, labels)