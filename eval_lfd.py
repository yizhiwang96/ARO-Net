
import os
import trimesh
import numpy as np
from joblib import Parallel, delayed
from lfd import LightFieldDistance
from options import get_parser
from trimesh import creation, transformations
import math


def cal_metrics(path_ref, path_rec, shape_id, dir_metrics, path_qry, path_occ, name_dataset):
    # rest of code
    mesh_1 = trimesh.load(path_ref)
    mesh_2 = trimesh.load(path_rec)

    lfd = LightFieldDistance(verbose=True).get_distance(
        mesh_1.vertices, mesh_1.faces,
        mesh_2.vertices, mesh_2.faces
    )

    fout = open(os.path.join(dir_metrics, shape_id + '.txt'), 'w')
    fout.write(str(lfd))
    fout.close()

def get_data_paths(args, is_ref=True):
    path_shapes = []
    path_qrys = []
    path_occs = []
    dir_dataset = os.path.join(args.dir_data, args.name_dataset)
    if args.name_dataset == 'shapenet':
        categories = args.categories_test.split(',')[:-1]
        fext_mesh = 'obj'
    else:
        categories = ['']
        if is_ref:
            fext_mesh = 'ply'
        else:
            fext_mesh = 'obj'
    
    id_shapes_all = []

    for category in categories:
        id_shapes = open(f'{dir_dataset}/04_splits/{category}/test.lst').read().split()
        id_shapes_all += id_shapes
        for shape_id in id_shapes:
            if is_ref:
                dir_mesh = os.path.join(dir_dataset, '00_meshes', category)
                path_shapes.append(os.path.join(dir_mesh, shape_id + '.' + fext_mesh))
                path_qrys.append(f'{dir_dataset}/02_qry_pts/{category}/{shape_id}.npy')
                if args.name_dataset == 'shapenet':
                    path_occs.append(f'{dir_dataset}/03_qry_occs/{category}/{shape_id}.npy')
                else:
                    path_occs.append(f'{dir_dataset}/03_qry_dists/{category}/{shape_id}.npy')
            else:
                dir_mesh = os.path.join('experiments', args.name_exp, 'results', args.name_dataset)
                path_shapes.append(os.path.join(dir_mesh, shape_id + '.' + fext_mesh))


    return path_shapes, id_shapes_all, path_qrys, path_occs

def eval(args):
    dir_metrics = os.path.join('./experiments', args.name_exp, 'metrics', args.name_dataset)
    os.makedirs(dir_metrics, exist_ok=True)

    path_shapes_ref, id_shapes, path_qrys, path_occs = get_data_paths(args, is_ref=True)
    path_shapes_rec, _, _, _ = get_data_paths(args, is_ref=False)


    assert(len(path_shapes_ref) == len(path_shapes_rec))

    '''
    # single-thread
    for idx, test_id in enumerate(id_shapes):
        cal_metrics(path_shapes_ref[idx], path_shapes_rec[idx], id_shapes[idx], dir_metrics, path_qrys[idx], path_occs[idx], args.name_dataset)
    '''

    tasks = []
    for idx, test_id in enumerate(id_shapes):
        tasks.append([path_shapes_ref[idx], path_shapes_rec[idx], id_shapes[idx], dir_metrics, path_qrys[idx], path_occs[idx], args.name_dataset])

    for task in tasks:
        cal_metrics(path_ref=task[0], path_rec=task[1], shape_id=task[2], dir_metrics=task[3], path_qry=task[4], path_occ=task[5], name_dataset=task[6])
    
def report(args):
    dir_metrics = os.path.join('./experiments', args.name_exp, 'metrics', args.name_dataset)
    path_shapes_rec, id_shapes, _, _ = get_data_paths(args, is_ref=False)

    name_metrics = ["lfd"]
    n_metrics = len(name_metrics)
    ret = np.zeros(n_metrics)

    for shape_id in id_shapes:
        content = open(os.path.join(dir_metrics, shape_id + '.txt')).read()
        tmp = content.split(' ')
        for idx in range(n_metrics):
            ret[idx] += float(tmp[idx])

    for idx in range(n_metrics):
        print(name_metrics[idx] + ': ' + str(ret[idx] / len(id_shapes)))

    return

if __name__ == "__main__":
    args = get_parser().parse_args()
    eval(args)
    report(args)