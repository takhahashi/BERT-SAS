import numpy as np
from sklearn.metrics import roc_auc_score

def calc_rcc_auc(true, pred, conf, metric_type, upper_score, reg_or_class=None, binary_risk=None):
  if metric_type == 'normal':
    risk = calc_risk(pred, true, reg_or_class, upper_score, binary=binary_risk)
    auc, points_x, points_y = calc_rcc_auc_simple(conf, risk)
  elif metric_type == 'rmse':
    auc, points_x, points_y = calc_rcc_auc_rmse(pred, true, conf, upper_score, reg_or_class)
  elif metric_type == 'scaledrmse':
    auc, points_x, points_y = calc_rcc_auc_scaledrmse(pred, true, conf, upper_score, reg_or_class)
  else:
    raise ValueError("{} is not a valid value for metric_type".format(metric_type))
  return auc, points_x, points_y

def calc_rcc_auc_simple(conf, risk):
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

def calc_rcc_auc_rmse(pred, true, conf, upper_score, reg_or_class):
  if reg_or_class == 'reg':
    pred_org = np.round(pred * upper_score).astype('int32')
    true_org = np.round(true * upper_score).astype('int32')
  else:
    pred_org = pred.astype('int32')
    true_org = true.astype('int32')
  risk = (pred_org - true_org) ** 2
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
  else:
    pred_org = pred.astype('int32')
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

def calc_rpp(conf, risk):
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
  else:
    int_scores = pred.astype('int32')
    int_true = true.astype('int32')
  return roc_auc_score(int_scores == int_true, conf)

def calc_risk(pred, true, reg_or_class, upper_score=None, binary=False):
  if binary == True:
    if reg_or_class =='reg':
      int_scores = np.round(pred * upper_score).astype('int32')
      int_true = np.round(true * upper_score).astype('int32')
    else:
      int_scores = pred.astype('int32')
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
    