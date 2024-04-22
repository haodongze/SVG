import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
import itertools
import json

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from setproctitle import setproctitle
from bisect import bisect
from torch.nn import functional as F

from datetime import datetime
import numpy as np

from data.dataset import VisDialDataset
from visdial.encoders import Encoder
from visdial.decoders import Decoder
from visdial.model import EncoderDecoderModel
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint

from single_evaluation import Evaluation

class SVG_dense(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)

    np.random.seed(hparams.random_seed[0])
    torch.manual_seed(hparams.random_seed[0])
    torch.cuda.manual_seed_all(hparams.random_seed[0])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    self.device = (
      torch.device("cuda", self.hparams.gpu_ids[0])
      if self.hparams.gpu_ids[0] >= 0
      else torch.device("cpu")
    )
    setproctitle(hparams.dataset_version + '_' + hparams.model_name + '_' + str(hparams.random_seed[0]))

  # def _build_data_process(self):
  def _build_dataloader(self):
    # =============================================================================
    #   SETUP DATASET, DATALOADER
    # =============================================================================
    old_split = "train" if self.hparams.dataset_version == "0.9" else None
    self.train_dataset = VisDialDataset(
      self.hparams,
      overfit=self.hparams.overfit,
      split="train",
      old_split = old_split,
      sample_flag=True
    )

    collate_fn = None
    if "dan" in self.hparams.img_feature_type:
      collate_fn = self.train_dataset.collate_fn

    self.train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.hparams.train_batch_size,
      num_workers=self.hparams.cpu_workers,
      shuffle=True,
      drop_last=True,
      collate_fn=collate_fn,
    )
    self.valid_dataset = VisDialDataset(
      self.hparams,
      overfit=self.hparams.overfit,
      split="val",
      old_split=None
    )

    self.valid_dataloader = DataLoader(
      self.valid_dataset,
      batch_size=self.hparams.eval_batch_size,
      num_workers=self.hparams.cpu_workers,
      drop_last=False,
      collate_fn=collate_fn
    )

    ###load ndcg label list
    samplefile = open("/data1/dzhao/rva/data/visdial_1.0_train_dense_sample.json", 'r')
    self.sample = json.loads(samplefile.read())
    samplefile.close()
    self.ndcg_id_list = []
    for idx in range(len(self.sample)):
        self.ndcg_id_list.append(self.sample[idx]['image_id'])

    print("""
      # -------------------------------------------------------------------------
      #   DATALOADER FINISHED
      # -------------------------------------------------------------------------
      """)

  def _build_model(self):

    # =============================================================================
    #   MODEL : Encoder, Decoder
    # =============================================================================

    print('\t* Building model...')
    # Pass vocabulary to construct Embedding layer.
    encoder = Encoder(self.hparams, self.train_dataset.vocabulary)
    decoder = Decoder(self.hparams, self.train_dataset.vocabulary)

    print("Encoder: {}".format(self.hparams.encoder))
    print("Decoder: {}".format(self.hparams.decoder))

    # New: Initializing word_embed using GloVe
    if self.hparams.glove_npy != '':
      encoder.word_embed.weight.data = torch.from_numpy(np.load(self.hparams.glove_npy))
      print("Loaded glove vectors from {}".format(self.hparams.glove_npy))
    # Share word embedding between encoder and decoder.
    decoder.word_embed = encoder.word_embed

    # Wrap encoder and decoder in a model.
    self.model = EncoderDecoderModel(encoder, decoder)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.model.parameters())))
    self.model = self.model.to(self.device)
    # Use Multi-GPUs
    if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
      self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

    # =============================================================================
    #   CRITERION
    # =============================================================================
    if "disc" in self.hparams.decoder:
      self.criterion = nn.CrossEntropyLoss()
      self.criterion_bce = nn.BCEWithLogitsLoss()

    elif "gen" in self.hparams.decoder:
      self.criterion = nn.CrossEntropyLoss(ignore_index=self.train_dataset.vocabulary.PAD_INDEX)

    # Total Iterations -> for learning rate scheduler
    if self.hparams.training_splits == "trainval":
      self.iterations = (len(self.train_dataset) + len(self.valid_dataset)) // self.hparams.virtual_batch_size
    else:
      self.iterations = len(self.train_dataset) // self.hparams.virtual_batch_size

    # =============================================================================
    #   OPTIMIZER, SCHEDULER
    # =============================================================================

    def lr_lambda_fun(current_iteration: int) -> float:
      """Returns a learning rate multiplier.

      Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
      and then gets multiplied by `lr_gamma` every time a milestone is crossed.
      """
      current_epoch = float(current_iteration) / self.iterations
      if current_epoch <= self.hparams.warmup_epochs:
        alpha = current_epoch / float(self.hparams.warmup_epochs)
        return self.hparams.warmup_factor * (1.0 - alpha) + alpha
      else:
        # return_val = 1.0
        # if current_epoch >= self.hparams.lr_milestones[0] and current_epoch < self.hparams.lr_milestones2[0]:
        #   idx = bisect(self.hparams.lr_milestones, current_epoch)
        #   return_val = pow(self.hparams.lr_gamma, idx)
        # elif current_epoch >= self.hparams.lr_milestones2[0]:
        #   idx = bisect(self.hparams.lr_milestones2, current_epoch)
        #   return_val = self.hparams.lr_gamma * pow(self.hparams.lr_gamma2, idx)
        # return return_val
        idx = bisect(self.hparams.lr_milestones, current_epoch)
        return pow(self.hparams.lr_gamma, idx)

    if self.hparams.lr_scheduler == "LambdaLR":
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.initial_lr)
      self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda_fun)
    else:
      raise NotImplementedError

    print(
      """
      # -------------------------------------------------------------------------
      #   Model Build Finished
      # -------------------------------------------------------------------------
      """
    )

  def _setup_training(self):
    # if self.hparams.save_dirpath == 'checkpoints/':
    self.save_dirpath = os.path.join(self.hparams.root_dir, self.hparams.save_dirpath)
    self.summary_writer = SummaryWriter(self.save_dirpath)
    self.checkpoint_manager = CheckpointManager(
      self.model, self.optimizer, self.save_dirpath, hparams=self.hparams
    )

    # If loading from checkpoint, adjust start epoch and load parameters.
    if self.hparams.load_pthpath == "":
      self.start_epoch = 1
    else:
      # "path/to/checkpoint_xx.pth" -> xx
      # self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
      self.start_epoch = 1
      model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
      if isinstance(self.model, nn.DataParallel):
        self.model.module.load_state_dict(model_state_dict)
      else:
        self.model.load_state_dict(model_state_dict)
      # self.optimizer.load_state_dict(optimizer_state_dict)
      self.previous_model_path = self.hparams.load_pthpath
      print("Loaded model from {}".format(self.hparams.load_pthpath))

    print(
      """
      # -------------------------------------------------------------------------
      #   Setup Training Finished
      # -------------------------------------------------------------------------
      """
    )

  def _loss_fn(self, epoch, batch, output):
    target = (batch["ans_ind"] if "disc" in self.hparams.decoder else batch["ans_out"])
    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1).to(self.device))

    return batch_loss

  def get_1round_idx_batch_data(self, batch, rnd, idx):  ##to get 1 round data with batch_size = 1
      temp_train_batch = {}
      for key in batch:
          if key in ['img_feat']:
              temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2].to(self.device)
          elif key in ['ques', 'opt', 'ques_len', 'opt_len', 'ans_ind']:
              temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2][:, rnd].to(self.device)
          elif key in ['hist_len', 'hist']:
              temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2][:, :rnd + 1].to(self.device)
          else:
              pass
      return temp_train_batch

  def get_1round_idx_batch_data_forrva(self, batch, rnd, idx):  ##to get 1 round data with batch_size = 1
      temp_train_batch = {}
      for key in batch:
          if key in ['img_feat']:
              temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2].to(self.device)
          elif key in ['ans_ind']:
              temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2][:, rnd].to(self.device)
          elif key in ['ques', 'ques_len', 'hist_len', 'hist', 'opt', 'opt_len']:
              temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2][:, :rnd + 1].to(self.device)
          else:
              pass
      return temp_train_batch

  def train(self):

    self._build_dataloader()
    self._build_model()
    self._setup_training()

    # Evaluation Setup
    evaluation = Evaluation(self.hparams, model=self.model, split="val")

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    global_iteration_step = (self.start_epoch - 1) * self.iterations

    running_loss = 0.0  # New
    train_begin = datetime.utcnow()  # New
    print(
      """
      # -------------------------------------------------------------------------
      #   Model Train Starts (NEW)
      # -------------------------------------------------------------------------
      """
    )
    for epoch in range(self.start_epoch, self.hparams.num_epochs):
      self.model.train()
      # -------------------------------------------------------------------------
      #   ON EPOCH START  (combine dataloaders if training on train + val)
      # -------------------------------------------------------------------------
      if self.hparams.training_splits == "trainval":
        combined_dataloader = itertools.chain(self.train_dataloader, self.valid_dataloader)
      else:
        combined_dataloader = itertools.chain(self.train_dataloader)

      print(f"\nTraining for epoch {epoch}:", "Total Iter:", self.iterations)
      tqdm_batch_iterator = tqdm(combined_dataloader)
      accumulate_batch = 0 # taesun New
      loss_function = self.hparams.loss_function
      for i, batch in enumerate(tqdm_batch_iterator):
        # for key in batch:
        #     batch[key] = batch[key].to(device)
        ##### find the round
        batchsize = batch['img_ids'].shape[0]
        grad_dict = {}
        self.optimizer.zero_grad()

        for idx in range(int(batchsize / 2)):
            for b in range(2):  # here is because with the batch_size = 1 will raise error
                sample_idx = self.ndcg_id_list.index(batch['img_ids'][idx * 2 + b].item())
                final_round = self.sample[sample_idx]['round_id'] - 1
                rnd = final_round
                ##for 1 round
                #temp_train_batch = get_1round_idx_batch_data(batch, rnd, idx)
                #output = model(temp_train_batch)[b]  ## this is only for avoid bug, no other meanings
                ##for 10 round (rva)
                temp_train_batch = self.get_1round_idx_batch_data_forrva(batch, rnd, idx)
                output = self.model(temp_train_batch)[b][-1]
                ##end 10 round (rva)
                target = batch["ans_ind"][b, rnd].to(self.device)
                rs_score = self.sample[sample_idx]['relevance']
                cuda_device = output.device

                if loss_function == 'R0':  # R0 loss (distance)
                    batch_loss = 0 #set this for higher NDCG score
                    # batch_loss = self.criterion(output.view(-1, output.size(-1)),
                    #                        target.view(-1))  # this is to keep MRR, can be deleted
                    rs_score = torch.tensor(rs_score).to(cuda_device)
                    output_sig = torch.sigmoid(output)
                    batch_loss += torch.sum(torch.pow((output_sig - rs_score), 2))
                    batch_loss = batch_loss / (100 + 1)
                elif loss_function == 'R1':  # R1 loss (Weighted Softmax)
                    # batch_loss = 0
                    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
                    loss_num_count = 0
                    for rs_idx in range(len(rs_score)):
                        if rs_score[rs_idx] != 0:
                            batch_loss += rs_score[rs_idx] * self.criterion(output.view(-1, output.size(-1)),
                                                                       torch.tensor(rs_idx).to(cuda_device).view(-1))
                            loss_num_count += 1
                    if loss_num_count != 0:
                        batch_loss = batch_loss / (loss_num_count + 1)  # prevent count = 0
                elif loss_function == 'R2':  # R2 loss (Binary Sigmoid)
                    # batch_loss = 0
                    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
                    output_sig = torch.sigmoid(output)
                    for rs_idx in range(len(rs_score)):
                        a = rs_score[rs_idx]
                        s = output_sig[rs_idx]
                        batch_loss += (1 + a) * - (a * torch.log(s) + (1 - a) * torch.log(1 - s))
                    batch_loss = batch_loss / len(rs_score)
                elif loss_function == 'R3':  # R3 loss (Generalized Ranking)
                    # batch_loss = 0
                    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
                    rs_score = torch.tensor(rs_score).to(cuda_device)
                    exp_sum = torch.sum(torch.exp(output[[idx for idx in range(len(rs_score)) if rs_score[idx] < 1]]))
                    loss_num_count = 0
                    for rs_idx in range(len(rs_score)):  # for the candidate with relevance score 1
                        if rs_score[rs_idx] > 0.8:
                            exp_sum = exp_sum + torch.exp(output[rs_idx])
                            batch_loss += (-output[rs_idx] + torch.log(exp_sum))
                            loss_num_count += 1
                            exp_sum = exp_sum - torch.exp(output[rs_idx])
                    exp_sum_2 = torch.sum(
                        torch.exp(output[[idx for idx in range(len(rs_score)) if rs_score[idx] < 0.4]]))
                    for rs_idx in range(len(rs_score)):  # for the candidate with relevance score 0.5
                        if rs_score[rs_idx] < 0.8 and rs_score[rs_idx] > 0.4:
                            exp_sum_2 = exp_sum_2 + torch.exp(output[rs_idx])
                            batch_loss += (-output[rs_idx] + torch.log(exp_sum_2))
                            loss_num_count += 1
                            exp_sum_2 = exp_sum_2 - torch.exp(output[rs_idx])
                    batch_loss = batch_loss / (loss_num_count + 1)
                elif loss_function == 'R4':  # R4 loss (Normalized BCE (the newest one), better than R2 and stable than R3)
                    batch_loss = 0
                    # batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
                    output_sig = torch.sigmoid(output)
                    rs_score = torch.tensor(rs_score).to(cuda_device)
                    rs_score = F.normalize(rs_score.unsqueeze(0), p=1).squeeze(0)  # norm
                    max_rs_score = torch.max(rs_score)
                    for rs_idx in range(len(rs_score)):
                        a = rs_score[rs_idx]
                        s = output_sig[rs_idx]
                        if s != 1:  # s cannot be 1
                            batch_loss += - 20 * (a * torch.log(s) + (max_rs_score - a) * torch.log(1 - s))
                    batch_loss = batch_loss / len(rs_score)
                else:
                    rs_score = torch.tensor(rs_score).to(cuda_device)
                    batch_loss = self.criterion_bce(output, rs_score)

                ##end loss computation
                if batch_loss != 0:  # prevent batch loss = 0
                    batch_loss.backward()
                    # count_loss += batch_loss.data.cpu().numpy()

        accumulate_batch += batch["img_ids"].shape[0]
        if self.hparams.virtual_batch_size == accumulate_batch \
            or i == (len(self.train_dataset) // self.hparams.train_batch_size): # last batch

          self.optimizer.step()

          # --------------------------------------------------------------------
          #    Update running loss and decay learning rates
          # --------------------------------------------------------------------
          if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
          else:
            running_loss = batch_loss.item()

          self.optimizer.zero_grad()
          accumulate_batch = 0

          self.scheduler.step(global_iteration_step)

          global_iteration_step += 1
          # torch.cuda.empty_cache()
          description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
            datetime.utcnow() - train_begin,
            epoch,
            global_iteration_step, running_loss,
            self.optimizer.param_groups[0]['lr'])
          tqdm_batch_iterator.set_description(description)

          # tensorboard
          if global_iteration_step % self.hparams.tensorboard_step == 0:
            description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
              datetime.utcnow() - train_begin,
              epoch,
              global_iteration_step, running_loss,
              self.optimizer.param_groups[0]['lr'],
              )
            self._logger.info(description)
            # tensorboard writing scalar
            self.summary_writer.add_scalar(
              "train/loss", batch_loss, global_iteration_step
            )
            self.summary_writer.add_scalar(
              "train/lr", self.optimizer.param_groups[0]["lr"], global_iteration_step
            )

      # -------------------------------------------------------------------------
      #   ON EPOCH END  (checkpointing and validation)
      # -------------------------------------------------------------------------
      self.checkpoint_manager.step(epoch)
      self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
      self._logger.info(self.previous_model_path)

      if epoch < self.hparams.num_epochs - 1 and self.hparams.dataset_version == '0.9':
        continue

      torch.cuda.empty_cache()
      evaluation.run_evaluate(self.previous_model_path, global_iteration_step, self.summary_writer,
                              os.path.join(self.checkpoint_manager.ckpt_dirpath, "ranks_%d_valid.json" % epoch))
      torch.cuda.empty_cache()

    return self.previous_model_path