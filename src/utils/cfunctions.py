import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score, roc_auc_score

class ScaleDiffBalance:
  def __init__(self, num_tasks, priority=None, beta=1.):
    self.num_tasks = num_tasks
    if priority is not None:
        self.task_priority = np.asarray(priority)
    else:
        self.task_priority = np.full(self.num_tasks, 1/self.num_tasks)
    self.all_loss_log = []
    self.loss_log = []
    for _ in range(self.num_tasks):
       self.loss_log.append([])
    self.beta = beta
  
  def update(self, all_loss, *args):
    self.all_loss_log = np.append(self.all_loss_log, all_loss)
    for idx, loss in enumerate(args):
       self.loss_log[idx] = np.append(self.loss_log[idx], loss)
  
  def __call__(self, *args):
    scale_weights = self._calc_scale_weights()
    diff_weights = self._calc_diff_weights()
    alpha = self._calc_alpha(diff_weights)
    all_loss = 0
    for w_s, w_d, l in zip(scale_weights, diff_weights, args):
      all_loss += w_s * w_d * l
    if len(self.all_loss_log) < 1:
      pre_loss = 0
    else:
      pre_loss = self.all_loss_log[-1]
    return alpha * all_loss, scale_weights, diff_weights, alpha, pre_loss
  
  def _calc_scale_weights(self):
    w_lis = []
    if len(self.all_loss_log) < 1:
      w_lis = np.full(self.num_tasks, self.task_priority)
    else:
      for task_priority, each_task_loss_arr in zip(self.task_priority, self.loss_log):
         w_lis = np.append(w_lis, self.all_loss_log[-1]*task_priority/each_task_loss_arr[-1])
    return torch.tensor(w_lis).cuda()
  
  def _calc_diff_weights(self):
    w_lis = []
    if len(self.all_loss_log) < 2:
      w_lis = np.ones(self.num_tasks)
    else:
      for each_task_loss_arr in self.loss_log:
        w_lis = np.append(w_lis, (each_task_loss_arr[-1]/each_task_loss_arr[-2])/(self.all_loss_log[-1]/self.all_loss_log[-2]))
    return torch.tensor(w_lis**self.beta).cuda()
  
  def _calc_alpha(self, diff_weights):
    if len(self.all_loss_log) < 2:
      return torch.tensor(1.).cuda()
    else:
      return (1/torch.sum(torch.tensor(self.task_priority).cuda() * diff_weights)).cuda()
    
def regvarloss(y_true, y_pre_ave, y_pre_var):
    loss = torch.exp(-torch.flatten(y_pre_var))*torch.pow(y_true - torch.flatten(y_pre_ave), 2)/2 + torch.flatten(y_pre_var)/2
    loss = torch.sum(loss)
    return loss

def simple_collate_fn(list_of_data):
  pad_max_len = torch.tensor(0)
  for data in list_of_data:
    if(torch.count_nonzero(data['attention_mask']) > pad_max_len):
      pad_max_len = torch.count_nonzero(data['attention_mask'])
  in_ids, token_type, atten_mask, labels = [], [], [], []
  for data in list_of_data:
    in_ids.append(data['input_ids'][:pad_max_len])
    token_type.append(data['token_type_ids'][:pad_max_len])
    atten_mask.append(data['attention_mask'][:pad_max_len])
    labels.append(data['labels'])
  batched_tensor = {}
  batched_tensor['input_ids'] = torch.stack(in_ids)
  batched_tensor['token_type_ids'] = torch.stack(token_type)
  batched_tensor['attention_mask'] = torch.stack(atten_mask)
  batched_tensor['labels'] = torch.tensor(labels)
  return batched_tensor


class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path
    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分

        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.val_loss_min = -score
            print(f'first_score: {-self.best_score}.     Saving model ...')
            torch.save(model.state_dict(), self.path)
            #self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score <= self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する