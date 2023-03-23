import numpy as np
import torch
from torch.utils.data import Dataset
import os
import trimesh
from scipy.spatial.transform import Rotation
from .utils import load_mesh

class ARONetDataset(Dataset):
    def __init__(self, split, args) -> None:
        self.split = split
        self.n_anc = args.n_anc
        self.n_qry = args.n_qry
        self.use_dist_hit = args.use_dist_hit
        self.dir_dataset = os.path.join(args.dir_data, args.name_dataset)
        self.anc_0 = np.load(f'./{args.dir_data}/anchors/sphere{str(self.n_anc)}.npy')
        self.anc = np.concatenate([self.anc_0[i::3] / (2 ** i) for i in range(3)])
        self.name_dataset = args.name_dataset
        self.n_pts_train = args.n_pts_train
        self.n_pts_val = args.n_pts_val
        self.n_pts_test = args.n_pts_test
        self.gt_source = args.gt_source
        self.files = []
        if self.name_dataset == 'shapenet':
            if self.split in {'train', 'val'}:
                categories = args.categories_train.split(',')[:-1]
            else:
                categories = args.categories_test.split(',')[:-1]
            self.fext_mesh = 'obj'
        else:
            categories = ['']
            self.fext_mesh = 'ply'
        for category in categories:
            id_shapes = open(f'{self.dir_dataset}/04_splits/{category}/{split}.lst').read().split()
            for shape_id in id_shapes:
                self.files.append((category, shape_id))
        
    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        category, shape_id = self.files[index]
        # For shapenet dataset: when do training and validation, we use the pcd, qry, occ provided by occ-net or im-net; 
        # when do testing, we input the points sampled from shapenet original mesh
        if self.split == 'train':
            if self.name_dataset == 'shapenet':
                pcd = np.load(f'{self.dir_dataset}/01_pcds/{category}/occnet/{shape_id}.npy')
                np.random.seed()
                perm = np.random.permutation(len(pcd))[:self.n_pts_train]
                pcd = pcd[perm]
            else:
                mesh = load_mesh(f'{self.dir_dataset}/00_meshes/{category}/{shape_id}.{self.fext_mesh}')
                pcd = mesh.sample(self.n_pts_train)
        elif self.split == 'val':
            if self.name_dataset == 'shapenet':
                pcd = np.load(f'{self.dir_dataset}/01_pcds/{category}/occnet/{shape_id}.npy')
                np.random.seed(1234)
                perm = np.random.permutation(len(pcd))[:self.n_pts_val]
                pcd = pcd[perm]                
            else:            
                pcd = np.load(f'{self.dir_dataset}/01_pcds/{category}/{str(self.n_pts_val)}/{shape_id}.npy')
        else:
            pcd = np.load(f'{self.dir_dataset}/01_pcds/{category}/{str(self.n_pts_test)}/{shape_id}.npy')

        if self.name_dataset == 'shapenet':
            qry = np.load(f'{self.dir_dataset}/02_qry_pts_{self.gt_source}/{category}/{shape_id}.npy')
            occ = np.load(f'{self.dir_dataset}/03_qry_occs_{self.gt_source}/{category}/{shape_id}.npy')
            sdf = occ
        else:
            qry = np.load(f'{self.dir_dataset}/02_qry_pts/{category}/{shape_id}.npy')
            sdf = np.load(f'{self.dir_dataset}/03_qry_dists/{category}/{shape_id}.npy')
            occ = (sdf >= 0).astype(np.float32) # sdf >= 0 means occ = 1

        if self.use_dist_hit and self.split == 'train': dist_hit = np.load(f'{self.dir_dataset}/05_hit_dist/{category}/{shape_id}.npy')

        if self.split == 'train':
            np.random.seed()
            perm = np.random.permutation(len(qry))[:self.n_qry]
            qry = qry[perm]
            occ = occ[perm]
            sdf = sdf[perm]
            if self.use_dist_hit and self.split == 'train': dist_hit = dist_hit[:, perm]
        else:
            np.random.seed(1234)
            perm = np.random.permutation(len(qry))[:self.n_qry]
            qry = qry[perm]
            occ = occ[perm]
            sdf = sdf[perm]
            if self.use_dist_hit and self.split == 'train': dist_hit = dist_hit[:, perm]           

        feed_dict = {
            'pcd': torch.tensor(pcd).float(),
            'qry': torch.tensor(qry).float(),
            'anc': torch.tensor(self.anc).float(),
            'occ': torch.tensor(occ).float(),
            'sdf': torch.tensor(sdf).float(),
        }
        if self.use_dist_hit and self.split == 'train': feed_dict['dist_hit'] = torch.tensor(dist_hit).float()

        return feed_dict


class SingleShapeDataset(Dataset):
    def __init__(self, split, args):
        super().__init__()
        self.split = split
        self.n_anc = args.n_anc
        self.n_qry = args.n_qry
        self.shape = args.name_single
        self.mesh = trimesh.load(f'./{args.dir_data}/single/{self.shape}.ply')
        self.vox = trimesh.load(f'./{args.dir_data}/single/{self.shape}.binvox')
        self.anc_0 = np.load(f'./{args.dir_data}/anchors/sphere{str(self.n_anc)}.npy')
        self.anc = np.concatenate([self.anc_0[i::3] / (2 ** i) for i in range(3)])
        self.n_pts_train = args.n_pts_train
        self.n_pts_val = args.n_pts_val
        self.n_pts_test = args.n_pts_test
        if split=='train':
            self.n_pts = self.n_pts_train
        elif split=='test':
            self.n_pts = self.n_pts_test
        elif split=='val':
            self.n_pts = self.n_pts_val
    
    def __len__(self): 
        if self.split == 'train':
            return 400
        else:
            return 64

    def __getitem__(self, index):
        mesh = self.mesh.copy()
        R = np.eye(4)
        rot = Rotation.random()
        R[:3,:3] = rot.as_matrix()
        mesh.apply_transform(R)

        scale = np.random.uniform(0.5,1.5,3)
        mesh.apply_scale(scale)

        box = mesh.bounding_box
        l = box.scale
        c = box.centroid

        mesh.vertices = (mesh.vertices-c)/l

        bounds = mesh.bounds+0.5
        extents = mesh.extents
        margin = 1-extents
        t = -bounds[0]+np.random.random(3)*margin
        mesh.vertices = mesh.vertices+t

        N_near = self.n_qry//2
        N_far = self.n_qry-N_near
        pts_near = mesh.sample(N_near)+(np.random.rand(N_near,3)-0.5)/16
        pts_far = np.random.random([N_far,3])-0.5
        pts = np.concatenate([pts_near,pts_far])
        pts_before = rot.inv().apply(((pts-t)*l+c)/scale[None])
        occ = self.vox.is_filled(pts_before).astype('float32')

        pcd = mesh.sample(self.n_pts)

        feed_dict = {
            'pcd':torch.tensor(pcd).float(),
            'anc':torch.tensor(self.anc).float(),
            'qry':torch.tensor(pts).float(),
            'occ':torch.tensor(occ).float()
        }
        return feed_dict