import torch.nn as nn
import torch
import torch.nn.functional as F
from . import layers
from .pointnets import ResnetPointnet, ResnetPointnetCondBN
from .pos_enc import PositionalEncoding1D
import numpy as np

class ARONetModel(nn.Module):
    def __init__(self, n_anc, n_qry, n_local, cone_angle_th, tfm_pos_enc=True, cond_pn=True, use_dist_hit=False, pn_use_bn=True, pred_type='occ', norm_coord=False):
        super().__init__()
        self.n_anc = n_anc
        self.n_local = n_local
        self.n_qry = n_qry
        self.cone_angle_th = cone_angle_th
        self.cond_pn = cond_pn
        self.use_dist_hit = use_dist_hit
        self.pred_type = pred_type
        self.norm_coord = norm_coord
        if self.cond_pn:
            self.point_net = ResnetPointnetCondBN(dim=4, reduce=True)
        else:
            self.point_net = ResnetPointnet(dim=4, reduce=True, size_aux=(n_anc, n_local), use_bn=pn_use_bn)
        self.tfm_pos_enc = tfm_pos_enc
        if self.cond_pn:
            self.fc_cond_1 = nn.Sequential(
                nn.Conv1d(3, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
            self.fc_cond_2 = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
        self.fc_1 = nn.Sequential(
            nn.Conv1d(4 + 128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        if self.use_dist_hit:
            self.fc_dist_hit = nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        if self.tfm_pos_enc:
            self.pos_enc = PositionalEncoding1D(128)
        self.att_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.att_decoder = nn.TransformerEncoder(self.att_layer, num_layers=6)
        if self.pred_type == 'occ':
            self.fc_out = nn.Conv1d(self.n_anc * 128, 1, 1)
        else:
            self.fc_out = nn.Sequential(
                nn.Conv1d(128, 1, 1),
                nn.Tanh()
            )         

    def cast_cone(self, pcd, anc, qry):
        """
        Using a cone to capture points in point cloud.
        Input:
        pcd: [n_bs, n_pts, 3]
        anc: [n_bs, n_anc, 3]
        qry: [n_bs, n_qry, 3]
        Return:
        hit_all: [n_bs, n_qry, n_anc, n_local, 3]       
        """
        top_k = self.n_local
        th = np.pi / (self.cone_angle_th)

        vec_anc2qry = qry[:, None, :, :] - anc[:, :, None, :]
        mod_anc2qry = torch.linalg.norm(vec_anc2qry, axis=-1)[:, :, :, None]
        norm_anc2qry = vec_anc2qry / mod_anc2qry
        ray_anc2qry = torch.cat([anc[:, :, None, :].expand(-1, -1, self.n_qry, -1), norm_anc2qry], -1)

        hit_all = []
        pcd_tile = pcd[:, None, :, :].expand(-1, self.n_qry, -1, -1)

        for idx_anc in range(self.n_anc):
            # first calculate the angle between anc2qry and anc2pts
            ray_anc2qry_ = ray_anc2qry[:, idx_anc, :, :]
            vec_anc2pts_ = pcd[:, None, :, :] - ray_anc2qry_[:, :, None, :3]
            mod_anc2pts_ = torch.linalg.norm(vec_anc2pts_, axis=-1)
            norm_anc2pts_ = vec_anc2pts_ / mod_anc2pts_[:, :, :, None]
            norm_anc2qry_ = ray_anc2qry_[:, :, None, 3:]
            cos = (norm_anc2qry_ * norm_anc2pts_).sum(-1)

            # filter out those points are not in the cone
            flt_angle = cos <= np.cos(th)
            mod_anc2pts_[flt_angle] = torch.inf
            tmp = torch.topk(mod_anc2pts_, top_k, dim=-1, largest=False)
            idx_topk, vl_topk = tmp.indices, tmp.values
            hit_raw = torch.gather(pcd_tile, 2, idx_topk[:, :, :, None].expand(-1, -1, -1, 3))
            flt_pts = (vl_topk == torch.inf).float()

            # padding those filtered-out points with query pints
            qry_rs = qry.unsqueeze(2).expand(-1, -1, top_k, -1)
            flt_pts_rs = flt_pts.unsqueeze(3).expand(-1, -1, -1, 3)
            hit = qry_rs * flt_pts_rs + hit_raw * (1 - flt_pts_rs)
            hit_all.append(hit.unsqueeze(2))

        hit_all = torch.cat(hit_all, 2)

        return hit_all

    def cal_relatives(self, hit, anc, qry):
        """
        Calculate the modulus and normals (if needed) from anc to query and query to hit points
        Input:
        hit: [n_bs, n_qry, n_anc, n_local, 3]
        anc: [n_bs, n_anc, 3]
        qry: [n_bs, n_qry, 3]
        Output:
        feat_anc2qry: [n_bs, n_qry, n_anc, 4]
        feat_qry2hit: [n_bs, n_qry, n_anc, n_local, 4]
        """

        vec_anc2qry = qry[:, :, None, :] - anc[:, None, :, :]
        mod_anc2qry = torch.linalg.norm(vec_anc2qry, axis=-1)[..., None]
        norm_anc2qry = torch.div(vec_anc2qry, mod_anc2qry.expand(-1, -1, -1, 3))
        if self.norm_coord == True:
            feat_anc2qry = torch.cat([norm_anc2qry, mod_anc2qry], -1)
        else:
            feat_anc2qry = torch.cat([vec_anc2qry, mod_anc2qry], -1)
        vec_qry2hit = hit - qry.view(-1, self.n_qry, 1, 1, 3)
        mod_qry2hit = torch.linalg.norm(vec_qry2hit, axis=-1)[..., None]
        mask_padded = (mod_qry2hit.expand(-1, -1, -1, -1, 3) == 0).float()
        mod_qry2hit_ = mod_qry2hit * (1 - mask_padded) + 1. * mask_padded # avoiding divide by 0
        norm_qry2hit = torch.div(vec_qry2hit, mod_qry2hit_)
        if self.norm_coord == True:
            feat_qry2hit = torch.cat([norm_qry2hit, mod_qry2hit], -1)
        else:
            feat_qry2hit = torch.cat([vec_qry2hit, mod_qry2hit], -1)

        return feat_anc2qry, feat_qry2hit

    def forward(self, feed_dict):

        pcd, qry, anc = feed_dict['pcd'], feed_dict['qry'], feed_dict['anc']
        n_bs, n_qry  = qry.shape[0], qry.shape[1]
        self.n_qry = n_qry # when doing marching cube, the number of query points may change

        # cast cone to capture local points (hit), and calculate observations from query points
        hit = self.cast_cone(pcd, anc, qry)
        feat_anc2qry, feat_qry2hit = self.cal_relatives(hit, anc, qry)
        
        # run point net to calculate local features
        feat_qry2hit_rs = feat_qry2hit.view(n_bs * self.n_qry * self.n_anc, self.n_local, -1)
        
        if self.cond_pn:
            cond_anc = self.fc_cond_2(self.fc_cond_1(anc.permute(0, 2, 1)))
            cond_anc = cond_anc.permute(0, 2, 1).unsqueeze(1).expand(-1, self.n_qry, -1, -1).reshape(n_bs * self.n_qry * self.n_anc, -1)
            feat_local = self.point_net(feat_qry2hit_rs, cond_anc)
        else:
            feat_local = self.point_net(feat_qry2hit_rs)
        
        feat_local = feat_local.view(n_bs, self.n_qry, self.n_anc, -1)

        # concat dir and dist from anc and merge them
        feat_local_radial = torch.cat([feat_anc2qry.permute(0, 3, 2, 1), feat_local.permute(0, 3, 2, 1)], 1)
        x = self.fc_1(feat_local_radial.reshape(n_bs, -1, self.n_anc * self.n_qry))
        x = self.fc_2(x)
        x = x.view(n_bs, -1, self.n_anc, self.n_qry).permute(0, 3, 2, 1) # n_bs, -1, n_anc, n_qry
        x = x.reshape(n_bs * self.n_qry, self.n_anc, -1)


        # apply positional encoding
        if self.tfm_pos_enc:

            x = x + self.pos_enc(x)
        
        x = self.att_decoder(x)
        
        # output the predicted occupancy
        x1 = x.view(n_bs, self.n_qry, self.n_anc, -1)
        x2 = x1.view(n_bs, self.n_qry, self.n_anc * 128).permute(0, 2, 1)
        pred = self.fc_out(x2).view(n_bs, self.n_qry)

        ret_dict = {}
        
        if self.pred_type == 'occ':
            ret_dict['occ_pred'] = pred
        else:
            ret_dict['sdf_pred'] = pred

        if self.use_dist_hit:
            hit_dist_pred = self.fc_dist_hit(x1).squeeze(-1).transpose(1, 2)
            ret_dict['dist_hit_pred'] = hit_dist_pred
        
        return ret_dict