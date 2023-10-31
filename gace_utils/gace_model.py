import torch.nn as nn

class GACEModel(nn.Module):

    def __init__(self, cfg, ip_dim, cp_dim, target_dim):
        super(GACEModel, self).__init__()

        g_cfg = cfg.GACE
        
        self.H_I = nn.Sequential(
            nn.Conv2d(ip_dim, g_cfg.MODEL.H_I_HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_I_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(g_cfg.MODEL.H_I_HIDDEN_DIM, g_cfg.MODEL.H_I_HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_I_HIDDEN_DIM),
            nn.ReLU(),
            nn.Conv2d(g_cfg.MODEL.H_I_HIDDEN_DIM, g_cfg.MODEL.H_I_OUTPUT_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_I_OUTPUT_DIM),
            nn.ReLU()
        )

        self.H_C = nn.Sequential(
            nn.Conv2d(cp_dim + g_cfg.MODEL.H_I_OUTPUT_DIM,
                      g_cfg.MODEL.H_C_HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_C_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(g_cfg.MODEL.H_C_HIDDEN_DIM, g_cfg.MODEL.H_C_HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_C_HIDDEN_DIM),
            nn.ReLU(),
            nn.Conv2d(g_cfg.MODEL.H_C_HIDDEN_DIM, g_cfg.MODEL.H_C_OUTPUT_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_C_OUTPUT_DIM),
            nn.ReLU(),
            nn.MaxPool2d((g_cfg.MAX_NR_NEIGHBORS, 1))
        )

        self.H_F = nn.Sequential(
            nn.Conv2d(g_cfg.MODEL.H_I_OUTPUT_DIM + g_cfg.MODEL.H_C_OUTPUT_DIM, 
                      g_cfg.MODEL.H_F_HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_F_HIDDEN_DIM),
            nn.ReLU(),
            nn.Conv2d(g_cfg.MODEL.H_F_HIDDEN_DIM, g_cfg.MODEL.H_F_HIDDEN_DIM, kernel_size=1),
            nn.BatchNorm2d(g_cfg.MODEL.H_F_HIDDEN_DIM),
            nn.ReLU(),
            nn.Conv2d(g_cfg.MODEL.H_F_HIDDEN_DIM, target_dim, kernel_size=1),
            nn.BatchNorm2d(target_dim),
        )

