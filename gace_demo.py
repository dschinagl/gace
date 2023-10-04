import argparse
import glob
import random
import numpy as np
import torch
import datetime
import pickle

from pcdet.config import cfg, cfg_from_yaml_file

from gace_utils.gace import GACE

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/gace_demo.yaml', 
                        help='demo config file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='base detector model weights')
    parser.add_argument('--batch_size_dg', type=int, default=8, 
                        help='batch size for data generation (model inference)')
    parser.add_argument('--batch_size_gace', type=int, default=2048, 
                        help='batch size for GACE training')
    parser.add_argument('--workers', type=int, default=4, 
                        help='number of workers for dataloader')
    parser.add_argument('--gace_data_folder', type=str, default='data_gace/', 
                        help='folder for generated train/val data and model')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    
    args, cfg = parse_config()
    
    gace = GACE(args, cfg)

    # create or load gace training/validation data
    train_data_f, val_data_f, val_det_annos_f = gace.init_gace_data()

    # train gace model
    gace_model = gace.train_model(train_data_f)

    # evaluate model
    results_dict = gace.eval_model(gace_model, val_data_f, val_det_annos_f)

    output_folder = args.gace_data_folder
    now = datetime.datetime.now()
    output_model_file = output_folder + 'gace_model_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.pth'
    output_result_file = output_folder + 'gace_result_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'

    torch.save(gace_model, output_model_file)

    with open(output_result_file, 'wb') as f:
        pickle.dump(results_dict, f)

    return



if __name__ == '__main__':
    main()


