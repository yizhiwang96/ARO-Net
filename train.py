import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.datasets import ARONetDataset,SingleShapeDataset
from src.models import ARONetModel
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import shutil
import time
import glob
from options import get_parser
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

def cal_acc(x, gt, pred_type):
    if pred_type == 'occ':
        acc = ((x['occ_pred'].sigmoid() > 0.5) == (gt['occ'] > 0.5)).float().sum(dim=-1) / x['occ_pred'].shape[1]
    else:
        acc = ((x['sdf_pred']>=0) == (gt['sdf'] >=0)).float().sum(dim=-1) / x['sdf_pred'].shape[1]
    acc = acc.mean(-1)
    return acc

def cal_loss_pred(x, gt, pred_type):

    if pred_type == 'occ':
        loss_pred = F.binary_cross_entropy_with_logits(x['occ_pred'], gt['occ'])
    else:
        loss_pred = F.l1_loss(x['sdf_pred'], gt['sdf'])

    return loss_pred

def train_step(batch, model, opt, args):
    for key in batch: batch[key] = batch[key].cuda()
    opt.zero_grad()
    x = model(batch)

    loss_pred = cal_loss_pred(x, batch, args.pred_type)
    loss = loss_pred
    if args.use_dist_hit:
        loss_hit_dist = F.l1_loss(x['dist_hit_pred'], batch['dist_hit'])
        loss += loss_hit_dist
    else:
        loss_hit_dist = torch.zeros(1)

    loss.backward()
    opt.step()
    with torch.no_grad():
        acc = cal_acc(x, batch, args.pred_type)
    return loss_pred.item(), loss_hit_dist.item(), acc.item()

@torch.no_grad()
def val_step(model, val_loader, pred_type):
    avg_loss_pred = 0
    avg_acc  = 0
    ni = 0
    for batch in val_loader:
        for key in batch: batch[key] = batch[key].cuda()
        x = model(batch)

        loss_pred = cal_loss_pred(x, batch, pred_type)

        acc = cal_acc(x, batch, pred_type)

        avg_loss_pred = avg_loss_pred + loss_pred.item()
        avg_acc  = avg_acc  + acc.item()
        ni += 1
    avg_loss_pred /=ni
    avg_acc /= ni
    return avg_loss_pred, avg_acc

def backup_code(name_exp):
    os.makedirs(os.path.join('experiments', name_exp, 'code'), exist_ok=True)
    shutil.copy('src/models.py', os.path.join('experiments', name_exp, 'code', 'models.py') )
    shutil.copy('src/datasets.py', os.path.join('experiments', name_exp, 'code', 'datasets.py'))
    shutil.copy('src/pointnets.py', os.path.join('experiments', name_exp, 'code', 'pointnets.py'))
    shutil.copy('src/layers.py', os.path.join('experiments', name_exp, 'code', 'layers.py'))
    shutil.copy('./train.py', os.path.join('experiments', name_exp, 'code', 'train_occ.py'))
    shutil.copy('./options.py', os.path.join('experiments', name_exp, 'code', 'options.py'))

def train(args):

    name_exp = args.name_exp
    name_exp_stamp = name_exp
    # name_exp_stamp = str(time.time()) + '_' + name_exp
    os.makedirs(os.path.join('experiments', name_exp_stamp), exist_ok=True)
    backup_code(name_exp_stamp)

    # Dump options
    with open(os.path.join('experiments', name_exp_stamp, "opts.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(str(key) + ": " + str(value) + "\n")

    dir_ckpt = os.path.join('experiments', name_exp_stamp, 'ckpt')
    os.makedirs(dir_ckpt, exist_ok=True)

    writer = SummaryWriter(os.path.join('experiments', name_exp_stamp, 'log'))

    if args.name_dataset in ['abc','shapenet']:
        train_loader = DataLoader(ARONetDataset(split='train', args=args), shuffle=True, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)
        val_loader = DataLoader(ARONetDataset(split='val', args=args), shuffle=False, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)
    else:
        train_loader = DataLoader(SingleShapeDataset(split='train', args=args), shuffle=True, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)
        val_loader = DataLoader(SingleShapeDataset(split='val', args=args), shuffle=False, batch_size=args.n_bs, num_workers=args.n_wk, drop_last=True)

    model = ARONetModel(n_anc=args.n_anc, n_qry=args.n_qry, n_local=args.n_local, cone_angle_th=args.cone_angle_th, tfm_pos_enc=args.tfm_pos_enc, 
                        cond_pn=args.cond_pn, use_dist_hit=args.use_dist_hit, pn_use_bn=args.pn_use_bn, pred_type=args.pred_type, norm_coord=args.norm_coord)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)

    model.cuda()

    opt = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        fnames_ckpt = glob.glob(os.path.join(dir_ckpt, '*'))
        fname_ckpt_latest = max(fnames_ckpt, key=os.path.getctime)
        # path_ckpt = os.path.join(dir_ckpt, fname_ckpt_latest)
        ckpt = torch.load(fname_ckpt_latest)
        model.module.load_state_dict(ckpt['model']) 
        opt.load_state_dict(ckpt['opt'])
        epoch_latest = ckpt['n_epoch'] + 1
        n_iter = ckpt['n_iter']
        n_epoch = epoch_latest
    else:
        epoch_latest = 0
        n_iter = 0
        n_epoch = 0
    
    for i in range(epoch_latest, args.n_epochs):
        model.train()
        for batch in train_loader:
            loss_pred, loss_hit_dist, acc = train_step(batch, model, opt, args)
            if n_iter % args.freq_log == 0:
                print('[train] epcho:', n_epoch, ' ,iter:', n_iter," loss_pred:", loss_pred, " loss_hit_dist:", loss_hit_dist, " acc:", acc)
                writer.add_scalar('Loss/train', loss_pred, n_iter)
                writer.add_scalar('Acc/train', acc, n_iter)
                
            n_iter += 1

        if n_epoch % args.freq_ckpt == 0:
            model.eval()
            avg_loss_pred, avg_acc = val_step(model, val_loader, args.pred_type)
            writer.add_scalar('Loss/val', avg_loss_pred, n_iter)
            writer.add_scalar('Acc/val', avg_acc, n_iter)
            print('[val] epcho:', n_epoch,' ,iter:',n_iter," avg_loss_pred:",avg_loss_pred, " acc:",avg_acc)
            if args.multi_gpu:
                torch.save({'model':model.module.state_dict(), 'opt':opt.state_dict(), 'n_epoch':n_epoch, 'n_iter':n_iter}, f'{dir_ckpt}/{n_epoch}_{n_iter}_{avg_loss_pred:.4}_{avg_acc:.4}.ckpt')
            else:
                torch.save({'model':model.state_dict(), 'opt':opt.state_dict(), 'n_epoch':n_epoch, 'n_iter':n_iter}, f'{dir_ckpt}/{n_epoch}_{n_iter}_{avg_loss_pred:.4}_{avg_acc:.4}.ckpt')
        if n_epoch > 0 and n_epoch % args.freq_decay == 0:
            for g in opt.param_groups:
                g['lr'] = g['lr'] * args.weight_decay
        
        n_epoch += 1


def main():
    args = get_parser().parse_args()
    if args.mode == 'train':
        train(args)
    else:
        test(args)

main()

        
