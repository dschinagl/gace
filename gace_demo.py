import argparse
import torch
from datetime import datetime
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file

from gace_utils.gace_data import GACEDataset
from gace_utils.gace_utils import GACELogger, train_gace_model, evaluate_gace_model


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
    parser.add_argument('--gace_data_folder', type=str, default='gace_data/', 
                        help='folder for generated train/val data and model')
    parser.add_argument('--gace_output_folder', type=str, default='gace_output/',
                        help='folder for gace output')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    
    args, cfg = parse_config()
    
    args.gace_output_folder = Path(args.gace_output_folder) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.gace_output_folder.mkdir(parents=True, exist_ok=True)

    logger = GACELogger(args.gace_output_folder)
    logger.gace_info('Demo for Geometry Aware Confidence Enhancement (GACE)')
    logger.gace_info(f'Dataset:\t {cfg.DATA_CONFIG.DATASET}')
    logger.gace_info(f'Base-Detector:\t {cfg.MODEL.NAME}')
    logger.gace_info(f'Data Folder:\t {args.gace_data_folder}')
    logger.gace_info(f'Output Folder:\t {args.gace_output_folder}')

    gace_dataset_train = GACEDataset(args, cfg, logger, train=True)
    gace_dataset_val = GACEDataset(args, cfg, logger, train=False)
    
    logger.gace_info('Start training confidence enhancement model')
    gace_model = train_gace_model(gace_dataset_train, args, cfg, logger)
    
    logger.gace_info('Start evaluation with new confidence scores')
    result_str = evaluate_gace_model(gace_model, gace_dataset_val, args, cfg, logger)
    logger.gace_info('Evaluation Results including GACE:')
    logger.gace_info(result_str)

    gace_model_path = args.gace_output_folder / 'gace_model.pth'
    torch.save(gace_model, gace_model_path)
    logger.gace_info(f'GACE model saved to {gace_model_path}')

    return


if __name__ == '__main__':
    main()


