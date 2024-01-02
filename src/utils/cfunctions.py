import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score, roc_auc_score

def multiclass_metric_loss_fast(represent, target, margin=10, class_num=2, start_idx=1,
                                per_class_norm=False):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = []
    for class_idx in range(start_idx, class_num + start_idx):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)

    loss_intra = torch.FloatTensor([0]).to(represent.device)
    num_intra = 0
    loss_inter = torch.FloatTensor([0]).to(represent.device)
    num_inter = 0
    for i in range(class_num):
        curr_repr = represent[indices[i]]
        s_k = len(indices[i])
        triangle_matrix = torch.triu(
            (curr_repr.unsqueeze(1) - curr_repr).norm(2, dim=-1)
        )
        buf_loss = torch.sum(1 / dim * (triangle_matrix ** 2))
        if per_class_norm:
            loss_intra += buf_loss / np.max([(s_k ** 2 - s_k), 1]) * 2
        else:
            loss_intra += buf_loss
            num_intra += (curr_repr.shape[0] ** 2 - curr_repr.shape[0]) / 2
        for j in range(i + 1, class_num):
            repr_j = represent[indices[j]]
            s_q = len(indices[j])
            matrix = (curr_repr.unsqueeze(1) - repr_j).norm(2, dim=-1)
            inter_buf_loss = torch.sum(torch.clamp(margin - 1 / dim * (matrix ** 2), min=0))
            if per_class_norm:
                loss_inter += inter_buf_loss / np.max([(s_k * s_q), 1])
            else:
                loss_inter += inter_buf_loss
                num_inter += repr_j.shape[0] * curr_repr.shape[0]
    if num_intra > 0 and not(per_class_norm):
        loss_intra = loss_intra / num_intra
    if num_inter > 0 and not(per_class_norm):
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter


def compute_loss_cer(logits, labels, loss, lamb=0.5, unpad=False):
    """Computes regularization term for loss with CER
    """
    # here correctness is always 0 or 1
    if unpad:
        # NER case
        logits = logits[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    # suppose that -1 will works for ner and cls
    confidence, prediction = torch.softmax(logits, dim=-1).max(dim=-1)
    correctness = prediction == labels
    correct_confidence = torch.masked_select(confidence, correctness)
    wrong_confidence = torch.masked_select(confidence, ~correctness)
    regularizer = 0
    if unpad:
        # speed up for NER
        regularizer = torch.sum(
            torch.clamp(wrong_confidence.unsqueeze(1) - correct_confidence, min=0)
            ** 2
        )
    else:
        for cc in correct_confidence:
            for wc in wrong_confidence:
                regularizer += torch.clamp(wc - cc, min=0) ** 2
    loss += lamb * regularizer
    return loss


def compute_loss_metric(hiddens, labels, loss, num_labels,
                        margin=10, lamb_intra=0.5, lamb=0.5, unpad=False):
    """Computes regularization term for loss with Metric loss
    """
    if unpad:
        hiddens = hiddens[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    class_num = num_labels
    start_idx = 0
    # TODO: define represent, target and margin
    # Get only sentence representaions
    loss_intra, loss_inter = multiclass_metric_loss_fast(
        hiddens,
        labels,
        margin=margin,
        class_num=class_num,
        start_idx=start_idx,
    )
    loss_metric = lamb_intra * loss_intra[0] + lamb * loss_inter[0]
    loss += loss_metric
    return loss

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

def mix_loss1(y_trues, y_preds, logits, high, low, alpha): #  \frac{\|\hat{y}-y\|^2}{-\log{\hat{P}_{y}}}-\log{\hat{P}_{y}}
   mse_loss, cross_loss = 0, 0
   y_trues_org = np.round(torch.flatten(y_trues).to('cpu').detach().numpy().copy() * (high - low))
   probs = logits.softmax(dim=1)[list(range(len(y_trues_org))), y_trues_org]
   neg_ln_probs = -torch.log(probs)
   loss = alpha * (((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)/neg_ln_probs) + neg_ln_probs
   mse_loss = torch.sum((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   cross_loss = torch.sum(-torch.log(probs))
   loss = torch.sum(loss)
   return loss, mse_loss, cross_loss

def mix_loss2(y_trues, y_preds, logits, high, low, alpha): #  -\hat{P}_{y} + \hat{P}_{y}\|\hat{y}-y\|^2
   mse_loss, cross_loss = 0, 0
   y_trues_org = np.round(torch.flatten(y_trues).to('cpu').detach().numpy().copy() * (high - low))
   probs = logits.softmax(dim=1)[list(range(len(y_trues_org))), y_trues_org]
   loss = -probs + probs * ((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   mse_loss = torch.sum((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   cross_loss = torch.sum(-torch.log(probs))
   loss = torch.sum(loss)
   return loss, mse_loss, cross_loss

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
  if labels[0].shape  != torch.tensor(1).shape:
     batched_tensor['labels'] = torch.stack(labels)
  else:
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