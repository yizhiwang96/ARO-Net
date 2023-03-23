import numpy as np
import os
import argparse
from src.utils import fibonacci_sphere

def fibo(args):
    points = fibonacci_sphere(args.n_anc).astype('float32')
    return points

def uniform(args):
    def sample_a_point():
        d = 100
        while(d > 0.5):  
            p = np.random.rand(3) - 0.5 # [0, 1) -> [-0.5, 0,5]
            d = np.linalg.norm(p)
        return p
    points = np.zeros((args.n_anc, 3))
    for i in range(args.n_anc):
        p = sample_a_point()
        points[i] = p

    return points

def grid(args):
    n_grid = 5
    x = np.linspace(0, 1, num=n_grid, endpoint=True, retstep=False, dtype=None, axis=0)
    y = np.linspace(0, 1, num=n_grid, endpoint=True, retstep=False, dtype=None, axis=0)
    z = np.linspace(0, 1, num=n_grid, endpoint=True, retstep=False, dtype=None, axis=0)
    x -= 0.5
    y -= 0.5
    z -= 0.5
    cnt = 0
    cnt_not_insphere = 0
    anchors = np.zeros((n_grid ** 3, 3))
    anchors_not_insphere = np.zeros((n_grid ** 3, 3))
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                d = np.linalg.norm(np.array([x[i], y[j], z[k]]))
                if d <= 0.5:
                    anchors[cnt][0] = x[i]
                    anchors[cnt][1] = y[j]
                    anchors[cnt][2] = z[k]
                    cnt += 1
                else:
                    anchors_not_insphere[cnt_not_insphere][0] = x[i]
                    anchors_not_insphere[cnt_not_insphere][1] = y[j]
                    anchors_not_insphere[cnt_not_insphere][2] = z[k]
                    cnt_not_insphere += 1
    anchors = anchors[0:cnt]
    anchors_not_insphere = anchors_not_insphere[:cnt_not_insphere]

    if cnt >= args.n_anc:
        perm = np.random.permutation(cnt)[:n_anchor]
        anchors = anchors[perm]

    if cnt < args.n_anc:
        cnt_left = args.n_anc - cnt
        perm = np.random.permutation(cnt_not_insphere)[:cnt_left]
        anchors_pad = anchors_not_insphere[perm]
        anchors = np.concatenate([anchors, anchors_pad], 0)
    
    return anchors

def construct_anchors(args):
    if args.method == 'fibo':
        points = fibo(args)
        path_tgt = os.path.join(args.path_save, f'sphere{str(args.n_anc)}_test.npy')
    elif args.method == 'grid':
        points = grid(args)
        path_tgt = os.path.join(args.path_save, f'grid{str(args.n_anc)}_test.npy')
    else:
        points = uniform(args)
        path_tgt = os.path.join(args.path_save, f'uniform{str(args.n_anc)}_test.npy')
    
    np.save(path_tgt, points)
    return points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='fibo', choices=['fibo', 'grid', 'uniform'])
    parser.add_argument("--n_anc", type=int, default=48)
    parser.add_argument("--path_save", type=str, default='./data/anchors/')
    args = parser.parse_args()
    construct_anchors(args)
