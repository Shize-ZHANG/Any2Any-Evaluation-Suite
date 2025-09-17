#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NR3D-Q v3: 无参考3D质量评分（只用 T/G/S 三项）
总分 = 100 * (0.3*T + 0.4*G + 0.3* S)

- T（拓扑/完整性，点云含“表面性”与连通性、法向一致性）
- G（几何质量，含PCA曲率、多尺度法向稳定、法向局部一致性）
- S（采样均匀性，含kNN均匀度、离群比例、表面性权重）

Usage:
    python eval3d_v3.py /path/to/model.ply
    python eval3d_v3.py /path/to/model.off
"""

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree as KDTree

warnings.filterwarnings("ignore")

try:
    import open3d as o3d
    import trimesh
except Exception as e:
    print("请先安装依赖: pip install open3d trimesh numpy scipy\n错误: ", e)
    sys.exit(1)

# ------------------------- 基础工具 -------------------------

def clamp(x, a=0.0, b=1.0):
    return float(np.minimum(np.maximum(x, a), b))

def robust_cv(x):
    """
    IQR/median 的鲁棒“变异系数”，经 exp 映射到 [0,1]（变异越大->值越接近1）。
    后续通常用 (1 - robust_cv) 作为得分。
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    med = np.median(x)
    if med <= 1e-12:
        med = np.mean(np.abs(x)) + 1e-12
    rcv = (iqr / (abs(med) + 1e-12))
    return float(1.0 - math.exp(-rcv))

def unit_normalize(verts):
    """
    中心化并缩放到单位立方体；返回归一化坐标与对角线 D(=sqrt(3))。
    """
    v = np.asarray(verts, dtype=np.float64)
    mins, maxs = v.min(0), v.max(0)
    center = (mins + maxs) / 2.0
    extents = (maxs - mins)
    scale = float(extents.max())
    if scale < 1e-12:
        scale = 1.0
    v_norm = (v - center) / scale
    D = math.sqrt(3.0)
    return v_norm, D

def load_any(path):
    path = path.strip()
    if path.lower().endswith(".ply"):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            raise ValueError("点云为空。")
        pts, D = unit_normalize(pts)
        pcd.points = o3d.utility.Vector3dVector(pts)
        return {"type": "pcd", "pcd": pcd, "diag": D}
    else:
        tm = trimesh.load(path, force="mesh", process=False)
        if (tm.vertices is None) or (tm.faces is None) or len(tm.faces) == 0:
            raise ValueError("网格为空或没有面。")
        verts, D = unit_normalize(tm.vertices)
        mesh = trimesh.Trimesh(vertices=verts, faces=tm.faces, process=False)
        return {"type": "mesh", "mesh": mesh, "diag": D}

def knn_indices(points, k=16):
    tree = KDTree(points)
    idx = tree.query(points, k=k+1)[1][:, 1:]
    return idx, tree

def knn_stats(points, k=8):
    tree = KDTree(points)
    d, _ = tree.query(points, k=k+1)  # 包含自身
    d = d[:, 1:]
    return d.mean(axis=1)

def normals_from_o3d(pcd, k=32, voxel_frac=None):
    """
    估计点云法向；可选按包围盒对角线比例做体素下采样。
    voxel_frac: None/<=0 则不下采样；否则 voxel_size = max(voxel_frac * bbox_diag, 1e-6)
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return np.zeros((0,3), dtype=np.float64)
    mins, maxs = pts.min(0), pts.max(0)
    bbox_diag = float(np.linalg.norm(maxs - mins))
    if voxel_frac is not None and voxel_frac > 0 and bbox_diag > 0:
        voxel_size = max(voxel_frac * bbox_diag, 1e-6)
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd_ds.points) >= 8:
            pcd = pcd_ds
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(pcd.normals)

def pca_eigs_local(points, idx):
    """
    对每个点的邻域做 PCA，返回 (N,3) 排序后的特征值 [λ1>=λ2>=λ3>=0]
    """
    N = points.shape[0]
    lam = np.zeros((N,3), dtype=np.float64)
    for i in range(N):
        nb = points[idx[i]]
        C = np.cov(nb.T)
        w, _ = np.linalg.eigh(C)
        w = np.sort(np.maximum(w, 0.0))[::-1]
        lam[i] = w
    return lam

def intrinsic_dimension_lb(points, k=20):
    """
    Levina–Bickel MLE 内在维度估计。
    表面点云≈2；体积随机点≈3。
    """
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k+1)
    dists = dists[:, 1:] + 1e-12  # 排除自身
    logs = np.log(dists[:, -1][:, None] / dists[:, :-1])
    mles = (k - 1) / np.sum(logs, axis=1)
    mles = mles[np.isfinite(mles)]
    if mles.size == 0:
        return 3.0
    return float(np.clip(np.median(mles), 1.0, 5.0))

def chamfer_bi_dist(pts_a, pts_b):
    """
    返回 (mean_dist_a2b, mean_dist_b2a, p90_a2b, p90_b2a)，
    虽然 v3 不再使用，但保留工具函数以便扩展。
    """
    ta = KDTree(pts_b)
    tb = KDTree(pts_a)
    d1, _ = ta.query(pts_a, k=1)
    d2, _ = tb.query(pts_b, k=1)
    return float(np.mean(d1)), float(np.mean(d2)), float(np.percentile(d1, 90)), float(np.percentile(d2, 90))

# ------------------------- 网格工具 -------------------------

def valence_stats(mesh: trimesh.Trimesh):
    adj = mesh.vertex_adjacency_graph
    deg = np.array([d for _, d in adj.degree()], dtype=np.float64)
    return deg

def triangle_quality_score(mesh: trimesh.Trimesh):
    """
    单元质量: q = 2*sqrt(3)*A / sum(e^2) ∈ (0,1]，等边三角形=1
    """
    v = mesh.vertices
    f = mesh.faces
    if len(f) == 0:
        return 0.0
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 1]]
    c = v[f[:, 0]] - v[f[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(-c, a), axis=1)
    e2 = np.sum(a*a, axis=1) + np.sum(b*b, axis=1) + np.sum(c*c, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        q = 2.0 * math.sqrt(3.0) * A / (e2 + 1e-12)
    q = np.clip(q, 0.0, 1.0)
    return float(np.mean(q))

def dihedral_angles(mesh: trimesh.Trimesh):
    """
    返回相邻面的二面角（弧度），用于几何噪声/尖锐度统计。
    """
    try:
        fa = mesh.face_adjacency
        if fa is None or len(fa) == 0:
            return np.array([])
        fn = mesh.face_normals
        a, b = fa[:, 0], fa[:, 1]
        dot = np.sum(fn[a] * fn[b], axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        ang = np.arccos(dot)
        return np.abs(ang)
    except Exception:
        return np.array([])

# ------------------------- 评分结构 -------------------------

@dataclass
class Scores:
    T: float
    G: float
    S: float

# ======== T: 拓扑/完整性 ========

def topo_score_mesh(mesh: trimesh.Trimesh) -> float:
    # 闭合性
    watertight = 1.0 if mesh.is_watertight else 0.0
    # 非流形边比例（>2 面共享）
    try:
        counts = mesh.edges_unique_counts
        nonmanifold_ratio = float(np.mean(counts > 2.0))
    except Exception:
        nonmanifold_ratio = 0.0
    nm_score = 1.0 - clamp(nonmanifold_ratio, 0.0, 1.0)
    # 近似自交占比（抽样粗测）
    self_intersect_ratio = 0.0
    try:
        faces = mesh.faces
        nF = len(faces)
        sampleF = min(2000, nF)
        if sampleF > 0:
            idx = np.random.choice(nF, sampleF, replace=False)
            tree = mesh.triangles_tree
            hits = 0
            checked = 0
            bounds = mesh.triangles_tree.bounds
            for i in idx:
                cand = tree.intersection(bounds[i])
                cand = [c for c in cand if c != i]
                checked += 1
                if len(cand) > 0:
                    hits += 1
            self_intersect_ratio = clamp(hits / (checked + 1e-9), 0.0, 1.0)
    except Exception:
        self_intersect_ratio = 0.0
    si_score = 1.0 - self_intersect_ratio
    # 连通分量数
    try:
        comps = mesh.split(only_watertight=False)
        c = len(comps)
    except Exception:
        c = 1
    comp_score = clamp(1.0 - (c - 1) / 3.0, 0.0, 1.0)
    # 汇总
    T = 0.35 * watertight + 0.25 * nm_score + 0.20 * si_score + 0.20 * comp_score
    return float(clamp(T))

def topo_like_from_pcd(pcd: o3d.geometry.PointCloud) -> float:
    """
    点云 T：表面性(内在维度≈2)、连通性（避免断裂长边）、法向局部一致性。
    """
    pts = np.asarray(pcd.points)
    N = len(pts)
    if N < 8:
        return 0.0

    # (1) 连通性近似：kNN 最小距离的95分位阈值长边占比
    tree = KDTree(pts)
    d, _ = tree.query(pts, k=9)
    d = d[:, 1:]
    mind = d.min(axis=1)
    thr = np.percentile(mind, 95)
    r_long = float(np.mean(mind > thr))
    conn_score = 1.0 - clamp(r_long, 0.0, 1.0)

    # (2) 表面性：内在维度 ID（2.0理想，3.0糟糕）
    id_est = intrinsic_dimension_lb(pts, k=20)
    surface_like = clamp(1.0 - (id_est - 2.0) / 1.0, 0.0, 1.0)

    # (3) 局部法向一致性
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    n = normals_from_o3d(p, k=24, voxel_frac=None)
    if n.shape[0] == N:
        idx, _ = knn_indices(pts, k=12)
        cos_list = []
        for i in range(N):
            nb = n[idx[i]]
            vi = n[i]
            cos = np.abs((nb @ vi).clip(-1, 1))
            cos_list.append(np.mean(cos))
        nc = float(np.mean(cos_list))  # [0,1]
    else:
        nc = 0.5

    # 综合
    T = 0.45 * surface_like + 0.35 * conn_score + 0.20 * nc
    return float(clamp(T))

# ======== G: 几何质量 ========

def geom_score_mesh(mesh: trimesh.Trimesh) -> float:
    ang = dihedral_angles(mesh)
    rc = robust_cv(ang) if ang.size > 0 else 0.0          # 曲率/起伏的鲁棒变异
    curv_score = 1.0 - clamp(rc, 0.0, 1.0)
    sharp_ratio = float(np.mean(ang > (math.pi / 3.0))) if ang.size > 0 else 0.0  # >60°
    sharp_score = 1.0 - clamp(sharp_ratio, 0.0, 1.0)
    tri_q = triangle_quality_score(mesh)  # [0,1]
    G = 0.4 * curv_score + 0.3 * sharp_score + 0.3 * tri_q
    return float(clamp(G))

def geom_score_pcd(pcd: o3d.geometry.PointCloud) -> float:
    pts = np.asarray(pcd.points)
    N = len(pts)
    if N < 32:
        return 0.0

    # (1) 局部PCA曲率 λ3/(λ1+λ2+λ3) —— 表面越薄(近2D)，均值越小
    idx, _ = knn_indices(pts, k=32)
    lam = pca_eigs_local(pts, idx)  # [λ1>=λ2>=λ3]
    curv = lam[:, 2] / (lam.sum(axis=1) + 1e-12)
    curv_mean = float(np.mean(curv))
    # ≤0.02 ~ 高分；≥0.08 ~ 低分
    curv_score = clamp(1.0 - (curv_mean - 0.02) / 0.06, 0.0, 1.0)

    # (2) 多尺度法向稳定：k=16 vs k=64 的夹角分布
    p1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    p2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    n1 = normals_from_o3d(p1, k=16, voxel_frac=None)
    n2 = normals_from_o3d(p2, k=64, voxel_frac=None)
    if n1.shape[0] == N and n2.shape[0] == N:
        dot = np.sum(n1 * n2, axis=1).clip(-1, 1)
        ang = np.arccos(dot)
        sharp_ratio = float(np.mean(ang > (math.pi / 6.0)))  # >30°
        sharp_score = 1.0 - clamp(sharp_ratio, 0.0, 1.0)
    else:
        sharp_score = 0.5

    # (3) 法向局部一致性：k=24 基准
    n3 = normals_from_o3d(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), k=24, voxel_frac=None)
    if n3.shape[0] == N:
        idx12, _ = knn_indices(pts, k=12)
        cos_list = []
        for i in range(N):
            cos = np.abs((n3[idx12[i]] @ n3[i]).clip(-1, 1))
            cos_list.append(np.mean(cos))
        nc = float(np.mean(cos_list))  # [0,1]
    else:
        nc = 0.5

    G = 0.45 * curv_score + 0.35 * sharp_score + 0.20 * nc
    return float(clamp(G))

# ======== S: 采样均匀性 ========

def sampling_score_pcd(pcd: o3d.geometry.PointCloud) -> float:
    pts = np.asarray(pcd.points)
    N = len(pts)
    if N < 16:
        return 0.0
    mean_d = knn_stats(pts, k=8)
    cv = robust_cv(mean_d)
    cv_score = 1.0 - clamp(cv, 0.0, 1.0)
    # 离群点比例
    mu, sigma = np.mean(mean_d), (np.std(mean_d) + 1e-12)
    z = np.abs((mean_d - mu) / sigma)
    r_out = float(np.mean(z > 2.5))
    out_score = 1.0 - clamp(r_out, 0.0, 1.0)
    # 表面性（内在维度）
    id_est = intrinsic_dimension_lb(pts, k=20)
    surface_like = clamp(1.0 - (id_est - 2.0) / 1.0, 0.0, 1.0)
    S = 0.5 * cv_score + 0.25 * out_score + 0.25 * surface_like
    return float(clamp(S))

def sampling_score_mesh(mesh: trimesh.Trimesh) -> float:
    area = mesh.area_faces
    area_cv = robust_cv(area)
    area_score = 1.0 - clamp(area_cv, 0.0, 1.0)
    deg = valence_stats(mesh)
    val_cv = robust_cv(deg)
    val_score = 1.0 - clamp(val_cv, 0.0, 1.0)
    sp = mesh.sample(5000) if len(mesh.faces) > 0 else mesh.vertices
    mean_d = knn_stats(sp, k=8) if len(sp) >= 8 else np.array([0.0])
    knn_cv = robust_cv(mean_d)
    knn_score = 1.0 - clamp(knn_cv, 0.0, 1.0)
    mu, sigma = np.mean(mean_d), (np.std(mean_d) + 1e-12)
    z = np.abs((mean_d - mu) / sigma)
    r_out = float(np.mean(z > 2.5))
    out_score = 1.0 - clamp(r_out, 0.0, 1.0)
    S = 0.4 * area_score + 0.25 * val_score + 0.2 * knn_score + 0.15 * out_score
    return float(clamp(S))

# ======== 汇总 ========

@dataclass
class TGScores:
    T: float
    G: float
    S: float

def score_object(obj):
    if obj["type"] == "mesh":
        mesh: trimesh.Trimesh = obj["mesh"]
        T = topo_score_mesh(mesh)
        G = geom_score_mesh(mesh)
        S = sampling_score_mesh(mesh)
    else:
        pcd: o3d.geometry.PointCloud = obj["pcd"]
        T = topo_like_from_pcd(pcd)
        G = geom_score_pcd(pcd)
        S = sampling_score_pcd(pcd)
    return TGScores(T=T, G=G, S=S)

def total_score(scores: TGScores) -> float:
    return float(np.clip(100.0 * (0.3 * scores.T + 0.4 * scores.G + 0.3 * scores.S), 0.0, 100.0))

# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description="NR3D-Q v3: 无参考3D质量评分 (T/G/S 三项，无 R)")
    parser.add_argument("input", type=str, help="输入3D文件路径（.ply 或 .off/.obj/.stl等）")
    args = parser.parse_args()

    np.random.seed(42)

    try:
        obj = load_any(args.input)
    except Exception as e:
        print("加载失败：", e)
        sys.exit(2)

    scores = score_object(obj)
    total = total_score(scores)

    print(f"[T 拓扑/完整性]    : {scores.T:.3f}")
    print(f"[G 几何质量]      : {scores.G:.3f}")
    print(f"[S 采样均匀性]    : {scores.S:.3f}")
    print("-" * 36)
    print(f"NR3D-Q 总质量分数 : {total:.2f}")

if __name__ == "__main__":
    main()
