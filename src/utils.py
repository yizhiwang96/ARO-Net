import numpy as np 
import math
import trimesh
import open3d as o3d

def load_mesh(fn):
    mesh = trimesh.load(fn, force='mesh', skip_materials=True, maintain_order=True, process=False)
    return mesh

def create_mesh_o3d(v,f):
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v),
        o3d.utility.Vector3iVector(f))
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def create_raycast_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene

def cast_rays(scene,rays):
    rays_o3dt = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = scene.cast_rays(rays_o3dt)
    # dist
    hit_dists = hits['t_hit'].numpy() # real/inf
    # mask
    hit_geos = hits['geometry_ids'].numpy()
    hit_mask = hit_geos!=o3d.t.geometry.RaycastingScene.INVALID_ID
    # hit_ids = np.where(hit_mask)[0]
    hit_dists[~hit_mask] = 1.0
    rdf = np.full_like(hit_dists, 1.0, dtype='float32')
    mask = np.full_like(hit_dists, 0.0, dtype='float32')
    rdf[hit_mask] = hit_dists[hit_mask] 
    mask[hit_mask] = hit_mask[hit_mask]
    return rdf, mask

def fibonacci_sphere(n=48,offset=False):
    """Sample points on sphere using fibonacci spiral.

    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    :param int n: number of sample points, defaults to 48
    :param bool offset: set True to get more uniform samplings when n is large , defaults to False
    :return array: points samples
    """

    golden_ratio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / golden_ratio

    if offset:
        if n >= 600000:
            epsilon = 214
        elif n>= 400000:
            epsilon = 75
        elif n>= 11000:
            epsilon = 27
        elif n>= 890:
            epsilon = 10
        elif n>= 177:
            epsilon = 3.33
        elif n>= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
        phi = np.arccos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    else:
        phi = np.arccos(1 - 2*(i+0.5)/n)

    x = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],axis=-1)
    return x

