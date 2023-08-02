import trimesh
import os
import numpy as np

def load_mesh(file_in):
    mesh = trimesh.load(file_in, force='mesh', skip_materials=True, maintain_order=True, process=False)
    return mesh

def normalize_mesh(file_in, file_out):

    mesh = load_mesh(file_in)

    bounds = mesh.extents
    if bounds.min() == 0.0:
        return
    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    scale = 1.0 / mesh.scale
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    
    mesh.export(file_out, include_texture=False)


def sample_point_cloud(path_mesh, n_pts):

    mesh = load_mesh(path_mesh)
    pts = mesh.sample(n_pts)

    return np.array(pts)

if __name__ == '__main__':
    
    path_src = f'mesh.obj'
    path_dst = f'mesh_normalized.obj'
    normalize_mesh(path_src, path_dst)

    point_cloud = sample_point_cloud(path_dst, n_pts=2048)
    np.save('point_cloud.npy', point_cloud)