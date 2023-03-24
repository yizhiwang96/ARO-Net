
import os
import trimesh
import numpy as np
from joblib import Parallel, delayed
from src.utils_eval import eval_chamfer, eval_iou, eval_hausdoff
from options import get_parser
from trimesh import creation, transformations
import math



def cal_metrics(path_ref, path_rec, shape_id, dir_metrics, path_qry, path_occ, name_dataset):
    try:
        mesh_ref, pts_ref = sample(path_ref)
        mesh_rec, pts_rec = sample(path_rec)

        ret = eval_chamfer(pts_rec, pts_ref)

        qry = np.load(path_qry)
        if name_dataset == 'shapenet':
            occ_gt = np.load(path_occ)
        else:
            sdf = np.load(path_occ)
            occ_gt = (sdf >= 0).astype(np.float32) # sdf >= 0 means occ = 1

        iou = eval_iou(mesh_rec, qry, occ_gt)
        dist_hsdf_rec2ref, dist_hsdf_ref2rec, dist_hsdf_max = eval_hausdoff(pts_rec, pts_ref)

        fout = open(os.path.join(dir_metrics, shape_id + '.txt'), 'w')
        for idx in range(len(ret)):
            fout.write(str(ret[idx]) + ' ')
        fout.write(str(iou) + ' ')
        fout.write(str(dist_hsdf_rec2ref) + ' ' + str(dist_hsdf_ref2rec) + ' ' + str(dist_hsdf_max))

        fout.close()
    except:
        print("[Warning] Fail to evaluate:",path_rec)
    


def sample(mesh_file, num_samples=10000):
    mesh = trimesh.load(mesh_file, force='mesh', skip_materials=True, maintain_order=True, process=False)
    samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
    return mesh, samples


def sample_ndc(mesh_file, is_rec=False, num_samples=10000):
    mesh = trimesh.load(mesh_file,force='mesh',skip_materials=True,maintain_order=True,process=False)
    if is_rec:
        matrix = np.eye(4)
        matrix[:3, :3] *= 1 / 64.
        mesh.apply_transform(matrix)
    
    direction = [0, 1, 0]
    center = [0, 0, 0]
    rot_matrix = transformations.rotation_matrix(math.pi / 2, direction, center)
    mesh.apply_transform(rot_matrix)
    
    if is_rec:
        translating_matrix = transformations.translation_matrix([-0.5, -0.5, 0.5])
        mesh.apply_transform(translating_matrix) 

    direction = [0, 1, 0]
    center = [0, 0, 0]
    rot_matrix = transformations.rotation_matrix(-math.pi / 2, direction, center)
    mesh.apply_transform(rot_matrix)

    samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
    return mesh, samples

def get_data_paths(args, is_ref=True):
    path_shapes = []
    path_qrys = []
    path_occs = []
    dir_dataset = os.path.join(args.dir_data, args.name_dataset)
    if args.name_dataset == 'shapenet':
        categories = args.categories_test.split(',')[:-1]
        if is_ref:
            fext_mesh = 'obj'
        else:
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
                path_qrys.append(f'{dir_dataset}/02_qry_pts_imnet/{category}/{shape_id}.npy')
                if args.name_dataset == 'shapenet':
                    path_occs.append(f'{dir_dataset}/03_qry_occs_imnet/{category}/{shape_id}.npy')
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

    with Parallel(n_jobs=8) as p:
        p(delayed(cal_metrics)(path_ref=task[0], path_rec=task[1], shape_id=task[2], dir_metrics=task[3], path_qry=task[4], path_occ=task[5], name_dataset=task[6]) for task in tasks) # y-> x: rec to ref
    
def report(args):
    dir_metrics = os.path.join('./experiments', args.name_exp, 'metrics', args.name_dataset)
    path_shapes_rec, id_shapes, _, _ = get_data_paths(args, is_ref=False)

    name_metrics = ["chamfer_L1", "chamfer_L2", "fscore", "precision", "recall", "iou", 'hausdoff_rec2ref', 'hausdoff_ref2rec', 'dist_hsdf_max']
    n_metrics = len(name_metrics)
    n_valid = 0
    ret = np.zeros(n_metrics)

    for shape_id in id_shapes:
        try:
            content = open(os.path.join(dir_metrics, shape_id + '.txt')).read()
            tmp = content.split(' ')
            for idx in range(n_metrics):
                ret[idx] += float(tmp[idx])
            n_valid+=1
        except:
            print("[Warning] Fail to load:",os.path.join(dir_metrics, shape_id + '.txt'))

    print(args.name_exp,args.name_dataset,f'({n_valid}/{len(id_shapes)})')
    for idx in range(n_metrics):
        print(name_metrics[idx] + ': ' + str(ret[idx] / n_valid))

    return

if __name__ == "__main__":
    args = get_parser().parse_args()
    eval(args)
    report(args)