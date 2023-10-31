import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.utils.loss_utils import SigmoidFocalClassificationLoss
from gace_utils.gace_model import GACEModel


class GACELogger(object):

    def __init__(self, output_folder):

        self.logger = common_utils.create_logger(output_folder / 'gace_log.txt')

    def info(self, msg):
        self.logger.info(msg)

    def gace_info(self, msg):
        msg = f'[GACE]\t {msg}'
        self.logger.info(msg) 


def train_gace_model(dataset_train, args, cfg, logger):
    logger.gace_info(f'Train GACE model for {cfg.GACE.TRAIN.NUM_EPOCHS} epochs')

    ip_dim = dataset_train.get_ip_dim()
    cp_dim = dataset_train.get_cp_dim()
    target_dim = dataset_train.get_target_dim()
    gace_model = GACEModel(cfg, ip_dim, cp_dim, target_dim)
    gace_model.cuda()
    gace_model.train()

    sampler_train = BatchSampler(
        RandomSampler(dataset_train), 
        batch_size=args.batch_size_gace, 
        drop_last=False)

    def my_collate(batch):
        return batch[0]

    dataloader_train = DataLoader(dataset_train, sampler=sampler_train,
                                  collate_fn=my_collate, num_workers=args.workers)
    
    iou_l1_loss = torch.nn.L1Loss()
    sigmoid_focal_loss = SigmoidFocalClassificationLoss(gamma=cfg.GACE.TRAIN.SFL_GAMMA, 
                                                        alpha=cfg.GACE.TRAIN.SFL_ALPHA)

    optimizer = torch.optim.Adam(gace_model.parameters(), lr=cfg.GACE.TRAIN.LR)

    total_iterations = len(dataloader_train) * cfg.GACE.TRAIN.NUM_EPOCHS

    p_bar_desc = f'[GACE] Training ({cfg.GACE.TRAIN.NUM_EPOCHS}) Epochs)'
    progress_bar = tqdm(total=total_iterations, desc=p_bar_desc, leave=True, 
                        dynamic_ncols=True)

    for epoch_count in range(cfg.GACE.TRAIN.NUM_EPOCHS):
        for ip_data, cp_data, nb_ip_data, cat, iou in dataloader_train:
            ip_data = ip_data.cuda()
            cp_data = cp_data.cuda()
            nb_ip_data = nb_ip_data.cuda()
            cat = cat.cuda()
            iou = iou.cuda()

            # forward instance specific features of neighbors first
            f_n_I = gace_model.H_I(nb_ip_data)
                
            # forward instance specific features of current detection
            f_I = gace_model.H_I(ip_data)
                
            # forward context features
            f_n_C = gace_model.H_C(torch.cat([cp_data, f_n_I.detach()], dim=1))

            # merge instance-specific and context feature vectors
            gace_output = gace_model.H_F(torch.cat([f_I, f_n_C], dim=1))

            sf_loss = sigmoid_focal_loss(gace_output[:, 0, ...], cat[:, None, None], 
                                         torch.ones_like(cat)[:, None, None])
            sf_loss = sf_loss.mean()
                                            
            iou_loss = iou_l1_loss(torch.sigmoid(gace_output[:, 1, ...]).flatten(), iou)

            loss = sf_loss + cfg.GACE.TRAIN.IOU_LOSS_W * iou_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            progress_bar.update()
        
    progress_bar.close()

    return gace_model


def evaluate_gace_model(gace_model, dataset_val, args, cfg, logger):

    base_dataset = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size_dg,
        dist=False, workers=args.workers,
        logger=logger,
        training=False)[0]

    sampler_val = BatchSampler(SequentialSampler(dataset_val),
                               batch_size=args.batch_size_gace,
                               drop_last=False)

    def my_collate(batch):
        return batch[0]

    dataloader_val = DataLoader(dataset_val, sampler=sampler_val,
                                collate_fn=my_collate, num_workers=args.workers)

    gace_model.eval()

    total_iterations = len(dataloader_val)

    p_bar_desc = f'[GACE] Evaluation'
    progress_bar = tqdm(total=total_iterations, desc=p_bar_desc,
                        leave=True, dynamic_ncols=True)

    new_scores = np.zeros(len(dataset_val))
    det_count = 0

    for ip_data, cp_data, nb_ip_data, cat, iou in dataloader_val:

        ip_data = ip_data.cuda()
        cp_data = cp_data.cuda()
        nb_ip_data = nb_ip_data.cuda()
        cat = cat.cuda()
        iou = iou.cuda()

        with torch.no_grad():
            f_n_I = gace_model.H_I(nb_ip_data)
            f_I = gace_model.H_I(ip_data)
            f_n_C = gace_model.H_C(torch.cat([cp_data, f_n_I.detach()], dim=1))
            gace_output = gace_model.H_F(torch.cat([f_I, f_n_C], dim=1))
                
            scores = torch.sigmoid(gace_output[:, 0, ...])

        new_scores[det_count:det_count+scores.shape[0]] = scores.flatten().cpu().numpy()

        det_count += scores.shape[0]

        progress_bar.update()

    progress_bar.close()
    
    det_annos = dataset_val.get_det_annos()

    det_count = 0
    
    for da in det_annos:
        da['score'] = new_scores[det_count:det_count+da['score'].shape[0]]
        det_count += da['score'].shape[0]

    result_str, result_dict = base_dataset.evaluation(
        det_annos, cfg.CLASS_NAMES, eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_dir=None)

    return result_str


