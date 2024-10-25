from LBM import LBM3D
from rigid import *
from LDDMM import *
import taichi as ti
import scipy.stats as stats

use_big = False
ti.init(arch=ti.gpu)
mat3 = ti.math.mat3
vec3 = ti.math.vec3

torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
P0 = []
for i in range(8):
    name = "./data/shooting/seed{}0.pt".format(i) if not use_big else "./data/shooting/seed{}0_big.pt".format(i)
    P0.append(torch.load(name))
P0_stack = torch.stack(P0)
smesh_name = "./input/disk.ply" if not use_big else "./input/bigdisk.ply"
sigma = torch.tensor([.6], dtype=torchdtype, device=torchdeviceId)
VS_np, FS_np = pp3d.read_mesh(smesh_name)
VS = torch.from_numpy(VS_np)
FS = torch.from_numpy(FS_np)
q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
Kv = GaussKernel(sigma=sigma)


def shoot(p0):
    p, q = Shooting(p0, q0, Kv, nt=20)[-1]
    return q


nu = 1.5e-5
rho = 1.29
c_rho = rho
gravity = -9.8

rigid_body_mass = 0.0005
s = 3
data =[]
mesh_scale = vec3(s, s, s)
seed_mass_center = vec3(0, 0, 0)
seed_mass_ratio = .3
mesh_mass_center_global = vec3(0, 0, 0)
# translation = vec3(32, 32, 32)
translation = vec3(20)
mesh_init_amom = vec3(0, 0, 0)
mesh_init_vel = vec3(0, 0, 0)
euler = vec3(0, 0, 0)
length_z = 0.2
nx = ny = nz = 41
use_grav = True
# +++++++++++++++++++++++++++++++++++++++++++++++
data = []
dx = length_z / (nz - 1)
c_L = dx  # [C]
mesh_mass = rigid_body_mass / (c_rho * c_L * c_L * c_L)
if not use_grav:
    gravity = 0
c_T = 0.1e-3  # [C]
c_nu = c_L * c_L / c_T  # [C]
c_M = c_rho * c_L * c_L * c_L
gravity_star = gravity / (c_L / c_T / c_T)
nu_star = nu / c_nu
# +++++++++++++++++++++++++++++++++++++++++++++++
info = rigid_info(smesh_name, mesh_mass, mesh_mass_center_global, seed_mass_center, seed_mass_ratio, mesh_scale, translation,
                  mesh_init_amom, mesh_init_vel, gravity_star, euler)
lbm = LBM3D(nx, ny, nz, nu_star, info)

def compute_loss_xshifting(w, sigma_1=np.pi * 1e-3, interp=False):
    w_sum = w.sum() if interp else 1.
    w_uq = w.unsqueeze(-1).unsqueeze(-1)
    weight_sum = (w_uq * P0_stack).sum(dim=0) / w_sum
    q = shoot(weight_sum).detach().cpu()

    eulers = (np.random.rand(3) - 0.5) * sigma_1
    info.mesh_euler = vec3(eulers[0], eulers[1], eulers[2])
    info.mesh_init_vel = vec3(0, 0, 0)
    prob = (1 / sigma_1) ** 3

    lbm.init_simulation()
    lbm.rb.reinitialize_simulation(info, q)
    bad_sample = False
    t0 = time.time()
    for r in range(0, 10000):
        lbm.step()
        mc = lbm.rb.global_mass_center[None]
        if mc.norm() > 1000 or mc.z > 0:
            bad_sample = True
            break
    t1 = time.time()
    print("Computed loss in", t1 - t0, "seconds")
    loss = 0
    if bad_sample:
        print('bad sample mc.norm = ', mc.norm())
        loss = 999999
    else:
        loss = -lbm.rb.global_mass_center[None].x
    return loss, prob


def compute_loss_falling(w, sigma_1=np.pi * 1e-3, interp=False):
    w_sum = w.sum() if interp else 1.
    w_uq = w.unsqueeze(-1).unsqueeze(-1)
    weight_sum = (w_uq * P0_stack).sum(dim=0) / w_sum
    q = shoot(weight_sum).detach().cpu()

    eulers = (np.random.rand(3) - 0.5) * sigma_1
    info.mesh_euler = vec3(eulers[0], eulers[1], eulers[2])
    info.mesh_init_vel = vec3(0, 0, 0)
    prob = (1 / sigma_1) ** 3

    lbm.init_simulation()
    lbm.rb.reinitialize_simulation(info, q)
    bad_sample = False
    t0 = time.time()
    for r in range(0, 8000):
        lbm.step()
        mc = lbm.rb.global_mass_center[None]
        if mc.norm() > 1000 or mc.z > 0:
            bad_sample = True
            break
    t1 = time.time()
    print("Computed loss in", t1 - t0, "seconds")
    loss = 0
    if bad_sample:
        print('bad sample mc.norm = ', mc.norm())
        loss = 999999
    else:
        loss = -lbm.rb.global_mass_center[None].z
    return loss, prob

def compute_loss_rot(w, sigma_1=np.pi * 1e-3, interp=False):
    w_sum = w.sum() if interp else 1.
    w_uq = w.unsqueeze(-1).unsqueeze(-1)
    weight_sum = (w_uq * P0_stack).sum(dim=0) / w_sum
    q = shoot(weight_sum).detach().cpu()

    eulers = (np.random.rand(3) - 0.5) * sigma_1
    info.mesh_euler = vec3(eulers[0], eulers[1], eulers[2])
    info.mesh_init_vel = vec3(0, 0, 0)
    prob = (1 / sigma_1) ** 3

    lbm.init_simulation()
    lbm.rb.reinitialize_simulation(info, q)
    bad_sample = False
    t0 = time.time()
    for r in range(0, 8000):
        lbm.step()
        mc = lbm.rb.global_mass_center[None]
        if mc.norm() > 1000 or mc.z > 0:
            bad_sample = True
            break
    t1 = time.time()
    print("Computed loss in", t1 - t0, "seconds")
    loss = 0
    if bad_sample:
        print('bad sample mc.norm = ', mc.norm())
        loss = 999999
    else:
        loss = -abs(lbm.rb.ang_vel[None].z)*1e3
    return loss, prob


def compute_loss_boom2(w, sigma_1=np.pi * 1e-3, interp=False):
    info.mesh_init_amom = vec3(0, 1e3, 0)
    info.mesh_init_vel = vec3(10e-2, 0, 5e-2)
    info.mesh_euler = vec3(np.pi / 2, 0, 0)
    w_sum = w.sum() if interp else 1.
    w_uq = w.unsqueeze(-1).unsqueeze(-1)
    weight_sum = (w_uq * P0_stack).sum(dim=0) / w_sum
    q = shoot(weight_sum).detach().cpu()
    prob = 1
    mc = vec3(0)
    l_mid = 0
    lbm.init_simulation()
    lbm.rb.reinitialize_simulation(info, q)
    bad_sample = False
    t0 = time.time()
    for r in range(0, 8000):
        lbm.step()
        mc = lbm.rb.global_mass_center[None]
        if r == 4000:
            l_mid = ti.sqrt(mc.x ** 2 + mc.y ** 2)
        if mc.norm() > 1000:
            bad_sample = True
            break
    t1 = time.time()
    print("Computed loss in", t1 - t0, "seconds")
    loss = 0
    if bad_sample:
        print('bad sample mc.norm = ', mc.norm())
        loss = 999999
    else:
        loss = ti.sqrt(mc.x ** 2 + mc.y ** 2) - l_mid
    return loss, prob
