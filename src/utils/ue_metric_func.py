import numpy as np
from sklearn.metrics import roc_auc_score

def calc_rcc_auc(true, pred, conf, metric_type, upper_score=None):
  if type(true[0]) != np.int32 and type(pred[0]) != np.int32:
    raise ValueError(f'`{type(true[0])}` is not valid type')
  if metric_type == 'normal':
    risk = (true != pred).astype('int32')
  elif metric_type == 'rmse':
    risk = (pred - true) ** 2
  elif metric_type == 'scaledrmse':
    pred_scaled = pred/upper_score
    true_scaled = true/upper_score
    risk = (pred_scaled - true_scaled) ** 2
  else:
    raise ValueError("{} is not a valid value for metric_type".format(metric_type))
  n = len(conf)
  cr_pair = list(zip(conf, risk))
  cr_pair.sort(key=lambda x: x[0], reverse=True)
  cumulative_risk = [cr_pair[0][1]]
  for i in range(1, n):
    cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])
  points_x, points_y = [], []
  auc = 0
  for k in range(n):
    auc += cumulative_risk[k] / (1+k)
    points_x.append((1+k) / n)  # coverage
    points_y.append(cumulative_risk[k] / (1+k))  # current avg. risk
  return auc, points_x, points_y

def calc_rcc_auc_simple(true, pred, conf):
  if type(true[0]) != np.int32 and type(pred[0]) != np.int32:
    raise ValueError(f'`{type(true[0])}` is not valid type')
  risk = (true != pred).astype('int32')
  n = len(conf)
  cr_pair = list(zip(conf, risk))
  cr_pair.sort(key=lambda x: x[0], reverse=True)
  cumulative_risk = [cr_pair[0][1]]
  for i in range(1, n):
    cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])
  points_x, points_y = [], []
  auc = 0
  for k in range(n):
    auc += cumulative_risk[k] / (1+k)
    points_x.append((1+k) / n)  # coverage
    points_y.append(cumulative_risk[k] / (1+k))  # current avg. risk
  return auc, points_x, points_y

def calc_rcc_auc_rmse(pred, true, conf):
  if type(true[0]) != np.int32 and type(pred[0]) != np.int32:
    raise ValueError(f'`{type(true[0])}` is not valid type')
  risk = (pred - true) ** 2
  n = len(conf)
  cr_pair = list(zip(conf, risk))
  cr_pair.sort(key=lambda x: x[0], reverse=True)
  cumulative_risk = [cr_pair[0][1]]
  for i in range(1, n):
    cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])
  points_x, points_y = [], []
  auc = 0
  for k in range(n):
    auc += cumulative_risk[k] / (1+k)
    points_x.append((1+k) / n)  # coverage
    points_y.append(cumulative_risk[k] / (1+k))  # current avg. risk
  return auc, points_x, points_y

def calc_rcc_auc_scaledrmse(pred, true, conf, upper_score, reg_or_class):
  if reg_or_class == 'reg':
    pred_org = np.round(pred * upper_score).astype('int32')
    true_org = np.round(true * upper_score).astype('int32')
  elif reg_or_class == 'class':
    pred_org = pred.astype('int32')
    true_org = true.astype('int32')
  elif reg_or_class == 'gp':
    pred_org = np.round(pred).astype('int32')
    true_org = true.astype('int32')

  pred_scaled = pred_org/upper_score
  true_scaled = true_org/upper_score
  risk = (pred_scaled - true_scaled) ** 2
  n = len(conf)
  cr_pair = list(zip(conf, risk))
  cr_pair.sort(key=lambda x: x[0], reverse=True)
  cumulative_risk = [cr_pair[0][1]]
  for i in range(1, n):
    cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])
  points_x, points_y = [], []
  auc = 0
  for k in range(n):
    auc += cumulative_risk[k] / (1+k)
    points_x.append((1+k) / n)  # coverage
    points_y.append(cumulative_risk[k] / (1+k))  # current avg. risk
  return auc, points_x, points_y

def calc_rpp(true: np.ndarray, pred: np.ndarray, uncert: np.ndarray):
  if type(true[0]) != np.int32 and type(pred[0]) != np.int32:
    raise ValueError(f'`{type(true[0])}` is not valid type')
  risk = (true != pred).astype('int32')
  conf = -uncert
  n = len(conf)
  cr_pair = list(zip(conf, risk))
  cr_pair.sort(key=lambda x: x[0], reverse=False)
  cnt = 0
  for i in range(n):
    for j in range(i, n):
      if(cr_pair[i][1] < cr_pair[j][1]):
        cnt += 1
  return cnt / (n**2)

def calc_roc_auc(pred, true, conf, reg_or_class, upper_score=None):
  if reg_or_class == 'reg':
    int_scores = np.round(pred * upper_score).astype('int32')
    int_true = np.round(true * upper_score).astype('int32')
  elif reg_or_class == 'class':
    int_scores = pred.astype('int32')
    int_true = true.astype('int32')
  elif reg_or_class == 'gp':
    int_scores = np.round(pred).astype('int32')
    int_true = true.astype('int32')
  return roc_auc_score(int_scores == int_true, conf)

def calc_risk(pred, true, reg_or_class, upper_score=None, binary=False):
  if binary == True:
    if reg_or_class =='reg':
      int_scores = np.round(pred * upper_score).astype('int32')
      int_true = np.round(true * upper_score).astype('int32')
    elif reg_or_class == 'class':
      int_scores = pred.astype('int32')
      int_true = true.astype('int32')
    elif reg_or_class == 'gp':
      int_scores = np.round(pred).astype('int32')
      int_true = true.astype('int32')
    return (int_scores != int_true).astype('int32')
  else:
    if reg_or_class =='reg':
      int_scores = np.round(pred * upper_score).astype('int32')
      int_true = np.round(true * upper_score).astype('int32')
    else:
      int_scores = pred.astype('int32')
      int_true = true.astype('int32')
    return (int_scores - int_true) ** 2
    