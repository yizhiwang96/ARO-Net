import numpy as np
import os
from src.utils import load_mesh, create_mesh_o3d, create_raycast_scene, cast_rays
from options import get_parser
from joblib import Parallel, delayed

class HitDistCalculator():
    def __init__(self, args) -> None:
        self.n_anc = args.n_anc
        self.n_qry = 24576
        self.dir_dataset = os.path.join(args.dir_data, args.name_dataset)
        self.anc_0 = np.load(f'./{args.dir_data}/anchors/sphere{str(self.n_anc)}.npy')
        self.anc = np.concatenate([self.anc_0[i::3] / (2 ** i) for i in range(3)])
        self.name_dataset = args.name_dataset
        self.files = []
        if self.name_dataset == 'shapenet':
            categories = args.categories_train.split(',')[:-1]
            self.fext_mesh = 'obj'
        else:
            categories = ['']
            self.fext_mesh = 'ply'
        
        for split in {'train', 'val', 'test'}:
            for category in categories:
                os.makedirs(f'{self.dir_dataset}/05_hit_dist/{category}/', exist_ok=True)
                id_shapes = open(f'{self.dir_dataset}/04_splits/{category}/{split}.lst').read().split()
                for shape_id in id_shapes:
                    self.files.append((category, shape_id))

    def cal_hit_distance(self, category, shape_id):

        mesh = load_mesh(f'{self.dir_dataset}/00_meshes/{category}/{shape_id}.{self.fext_mesh}')
        qry = np.load(f'{self.dir_dataset}/02_qry_pts/{category}/{shape_id}.npy')

        vec_anc2pts = qry[:, None, :] - self.anc[None, :, :]
        mod_anc2pts = np.linalg.norm(vec_anc2pts, axis=-1, keepdims=True)
        norm_anc2pts = vec_anc2pts / mod_anc2pts
        ray_anc2pts = []
        for i in range(self.n_anc):
            origin =  self.anc[i]
            ray_anc2pts_ = np.concatenate([origin + np.zeros_like(norm_anc2pts[:, i, :]), norm_anc2pts[:, i, :]], axis=-1)
            ray_anc2pts.append(ray_anc2pts_) 
        ray_anc2pts = np.concatenate(ray_anc2pts)
        mesh_o3d = create_mesh_o3d(mesh.vertices, mesh.faces)
        scene = create_raycast_scene(mesh_o3d)
        dist_hit, _ = cast_rays(scene, ray_anc2pts)
        dist_hit = dist_hit.astype('float32').reshape(self.n_anc, self.n_qry)
        
        np.save(f'{self.dir_dataset}/05_hit_dist/{category}/{shape_id}.npy', dist_hit)

    def cal_multi_processes(self):
        
        with Parallel(n_jobs=8) as p:
            p(delayed(self.cal_hit_distance)(category, shape_id) for category, shape_id in self.files)


if __name__ == "__main__":
    args = get_parser().parse_args()
    calculator = HitDistCalculator(args)
    calculator.cal_multi_processes()