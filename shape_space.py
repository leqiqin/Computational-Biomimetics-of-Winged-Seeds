import torch

from LDDMM import *
from LDDMM import *
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# import pykeops.numpy as pnp

P0 = []
qt = []
w0 = np.zeros(14)
w0.fill(0)
for i in range(14):
    name = "./data/shooting/seed{}0.pt".format(i)
    P0.append(torch.load(name))

smesh_name = "./input/disk.ply"
sigma = torch.tensor([.6], dtype=torchdtype, device=torchdeviceId)
VS_np, FS_np = pp3d.read_mesh(smesh_name)
# remesher = LDDMM_remesher(VS_np, FS_np)
VS = torch.from_numpy(VS_np)
FS = torch.from_numpy(FS_np)
q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
Kv = GaussKernel(sigma=sigma)
# Kv_solver = GaussKernel_Solver(sigma=sigma)
save_shooting = True
ps.init()
li = ps.register_surface_mesh("Linear interp", VS, FS_np)


def solve(p00, q00):
    p, q = Shooting(p00, q00, Kv, nt=20)[-1]
    return q


def solve_all(p00, q00):
    return Shooting(p00, q00, Kv, nt=20)


def flow(q00, pqlist, dt=1 / 20):
    qi_new = q00
    for nt in range(20):
        pi, qi = pqlist[nt]
        qi_new_dot = Kv(qi_new, qi, pi)
        pi_dot, qi_dot = HamiltonianSystem(Kv)(pi, qi)
        pi_mid = pi + (2 * dt / 3) * pi_dot
        qi_mid = qi + (2 * dt / 3) * qi_dot
        qi_new_mid = qi_new + (2 * dt / 3) * qi_new_dot
        qi_new_dot_mid = Kv(qi_new_mid, qi_mid, pi_mid)
        qi_new = qi_new + (0.25 * dt) * (qi_new_dot + 3 * qi_new_dot_mid)
    return qi_new


def GaussKernel_Solver(alpha, sigma, x, b):
    K_xx = torch.exp(
        -torch.sum((x[:, None, :] - x[None, :, :]) ** 2, dim=2) / (sigma ** 2)
    )
    c = torch.linalg.solve(K_xx, b)
    return c


def call_back():
    global w0, li, lc, rs, lcr
    changed, w0[0] = psim.SliderFloat("w1", w0[0], v_min=0, v_max=1)
    changed, w0[1] = psim.SliderFloat("w2", w0[1], v_min=0, v_max=1)
    changed, w0[2] = psim.SliderFloat("w3", w0[2], v_min=0, v_max=1)
    changed, w0[3] = psim.SliderFloat("w4", w0[3], v_min=0, v_max=1)
    changed, w0[4] = psim.SliderFloat("w5", w0[4], v_min=0, v_max=1)
    changed, w0[5] = psim.SliderFloat("w6", w0[5], v_min=0, v_max=1)
    changed, w0[6] = psim.SliderFloat("w7", w0[6], v_min=0, v_max=1)
    changed, w0[7] = psim.SliderFloat("w8", w0[7], v_min=0, v_max=1)
    changed, w0[8] = psim.SliderFloat("w9", w0[8], v_min=0, v_max=1)
    changed, w0[9] = psim.SliderFloat("w10", w0[9], v_min=0, v_max=1)
    changed, w0[10] = psim.SliderFloat("w11", w0[10], v_min=0, v_max=1)
    changed, w0[11] = psim.SliderFloat("w12", w0[11], v_min=0, v_max=1)
    changed, w0[12] = psim.SliderFloat("w13", w0[12], v_min=0, v_max=1)
    changed, w0[13] = psim.SliderFloat("w14", w0[13], v_min=0, v_max=1)
    # changed, w5 = psim.SliderFloat("w5", w5, v_min=0, v_max=1)
    if psim.Button("Random shooting"):
        p0 = torch.from_numpy(np.random.randn(5000, 3) * 1e-1).to(dtype=torchdtype,
                                                                  device=torchdeviceId).requires_grad_(True)
        q = solve(p0).detach().cpu()
        ps.get_surface_mesh("Shooting result").update_vertex_positions(q)

    if psim.Button("Shoot"):
        w = w0.sum()
        p0_this = w0[0] * P0[0] / w
        p0_this_1 = w0[0] * P0[0]
        for i in range(1, 14):
            p0_this += w0[i] * P0[i] / w
            p0_this_1 += w0[i] * P0[i]
        q = solve(p0_this, q0).detach().cpu()
        pqlist = solve_all(p0_this_1, q0)
        q1 = pqlist[-1][1].detach().cpu()
        li.update_vertex_positions(q)
        if save_shooting:
            pp3d.write_mesh(q, FS_np, "./data/shooting_interp.obj")
            pp3d.write_mesh(q1, FS_np, "./data/shooting_lincomb.obj")


ps.set_up_dir("z_up")
ps.set_ground_plane_mode("none")
ps.set_ground_plane_height_factor(1)
ps.set_user_callback(call_back)
ps.show()