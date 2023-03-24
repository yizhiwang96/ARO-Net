import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # dataset related
    parser.add_argument('--dir_data', type=str, default='./data')
    parser.add_argument('--name_dataset', type=str, default='abc', choices=['abc', 'shapenet', 'single', 'custom'])
    parser.add_argument('--name_single', type=str, default='fertility', help='name of the single shape')
    parser.add_argument('--n_wk', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--categories_train', type=str, default='03001627,', help='the training and validation categories of objects for ShapeNet datasets')
    parser.add_argument('--categories_test', type=str, default='03001627,02691156,', help='the testing categories of objects for ShapeNet datasets')
    parser.add_argument('--add_noise', type=float, default=0, help='the std of noise added to the point clouds')
    parser.add_argument('--gt_source', type=str, default='occnet', choices=['imnet', 'occnet'], help='using which query-occ groundtruth when training on Shapenet')
    # ARO-Net hyper-parameters 
    parser.add_argument('--n_pts_train', type=int, default=2048, help='the number of points sampled from a mesh when training')
    parser.add_argument('--n_pts_val', type=int, default=1024, help='the number of points of pcd when validation')
    parser.add_argument('--n_pts_test', type=int, default=1024, help='the number of points of pcd when testing')
    parser.add_argument('--cone_angle_th', type=float, default=15., help='parameter used to control the cone angle, which will be 2 * np.pi / (opts.cone_angle_th)')
    parser.add_argument('--n_local', type=int, default=16, help='the number of points will be captured in a cone')
    parser.add_argument('--n_anc', type=int, default=48, help='the number of anchors')
    parser.add_argument('--n_qry', type=int, default=512, help='the number of query points for per shape when training')
    parser.add_argument('--pn_use_bn', action=argparse.BooleanOptionalAction, help='using batch normalization for pointnet (not cond-bn-pointnet)')
    parser.add_argument('--cond_pn', action=argparse.BooleanOptionalAction, help='whether to use conditional pointnet')
    parser.add_argument('--use_dist_hit', action=argparse.BooleanOptionalAction, help='predict hit distance as an auxiliary loss, run cal_hit_dist.py first')
    parser.add_argument('--tfm_pos_enc', action=argparse.BooleanOptionalAction, help='using transformer postisitonal encoding')
    parser.add_argument('--pred_type', type=str, default='occ', choices=['occ', 'sdf'], help='predict occupancy (occ) or signed distance field (sdf), sdf only works for abc dataset')
    parser.add_argument('--norm_coord', action=argparse.BooleanOptionalAction, help='normalize the coordninates(modulus transformed into 1) before feeding them into networks')
    # common hyper-parameters
    parser.add_argument('--name_exp', type=str, default='20220322_shapenet_topk16_bs8_qry512_train2048test1024_useoccnetdata_normcoord')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--n_bs', type=int, default=8, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='init learning rate')
    parser.add_argument('--n_dim', type=int, default=128, help='the dimension of hidden layer features')
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--freq_ckpt', type=int, default=10, help='frequency of epoch saving checkpoint')
    parser.add_argument('--freq_log', type=int, default=200, help='frequency of outputing training logs')
    parser.add_argument('--freq_decay', type=int, default=100, help='decaying the lr evey freq_decay epochs')
    parser.add_argument('--weight_decay', type=float, default=0.5, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visualize loss')
    parser.add_argument('--resume', type=bool, default=False, help='resume training')
    # Marching Cube realted
    parser.add_argument('--mc_chunk_size', type=int, default=3000, help='the number of query points in a chunk when doing marching cube, set it according to your GPU memory')
    parser.add_argument('--mc_res0', type=int, default=64, help='start resolution for MISE')
    parser.add_argument('--mc_up_steps', type=int, default=2, help='number of upsampling steps')
    parser.add_argument('--mc_threshold', type=float, default=0.5, help='the threshold for network output values')
    # testing related
    parser.add_argument('--name_ckpt', type=str, default='10_5511_0.0876_0.9612.ckpt')
    
    return parser