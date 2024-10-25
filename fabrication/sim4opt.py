from LBM import LBM3D
from rigid import *
from LDDMM import *
import taichi as ti
import scipy.stats as stats
use_big = True
ti.init(arch=ti.gpu)
mat3 = ti.math.mat3
vec3 = ti.math.vec3

torchdeviceId = torch.device("cuda:0")
P0 = []
single_winged_index = [7, 8, 9, 11, 13]
for i in range(5):
    name = "../data/shooting/seed{}0_big.pt".format(single_winged_index[i])
    P0.append(torch.load(name))

smesh_name = "../input/disk.ply" if not use_big else "../input/bigdisk.ply"
# print(smesh_name)
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
c_rho = rho  # [C]
gravity = -9.8

rigid_body_mass = 0.00008
s = 2.2
mesh_scale = vec3(s, s, s)
seed_mass_center = vec3(-0.08, 0.12, 0.01)
seed_mass_ratio = 0.4
addweight_beginpoint = vec3(1.00, 4.60, 0.01)
addweight_endpoint = vec3(0.28, 0.28, 0.01)
addweight_diameter = 0.4e-3
addweight_mass_rho = 1.6 * 7.86e3 * addweight_diameter ** 2 * 0.25 * np.pi
mesh_mass_center_global = vec3(0, 0, 0)
translation = vec3(16, 16, 16)
mesh_init_amom = vec3(0, 0, 0)
mesh_init_vel = vec3(0, 0, 0)
euler = vec3(0, 0, 0)
length_z = 0.125
nx = ny = nz = 33
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
mass_coeff = 1 / (c_rho * c_L * c_L * c_L)
leaf_rho = 0.074
# +++++++++++++++++++++++++++++++++++++++++++++++
info = rigid_info(smesh_name, mesh_mass, mesh_mass_center_global, seed_mass_center, seed_mass_ratio, mesh_scale,
                  translation,
                  mesh_init_amom, mesh_init_vel, gravity_star, euler,
                  addweight_beginpoint, addweight_endpoint, addweight_mass_rho,
                  mass_coeff, leaf_rho, c_L)
lbm = LBM3D(nx, ny, nz, nu_star, info)

def compute_loss_rot(w, sigma_1=np.pi * 1e-3, interp=False):
    w_sum = 1
    if interp:
        w_sum = w.sum()
    p0_this = w[0] * P0[0] / w_sum
    for i in range(1, 5):
        p0_this += w[i] * P0[i] / w_sum
    q = shoot(p0_this).detach().cpu()

    eulers = (np.random.rand(3) - 0.5) * sigma_1
    info.mesh_euler = vec3(eulers[0], eulers[1], eulers[2])
    info.mesh_init_vel = vec3(0, 0, 0)
    prob = (1 / sigma_1) ** 3
    lbm.init_simulation()
    lbm.rb.reinitialize_simulation(info, q)
    bad_sample = False
    t0 = time.time()
    for r in range(0, 15000):
        lbm.step()
        mc = lbm.rb.global_mass_center[None]
        if mc.norm() > 4000 or mc.z > 0:
            bad_sample = True
            break
    t1 = time.time()
    print("Computed loss in", t1 - t0, "seconds")
    loss = 0
    if bad_sample:
        print('bad sample mc.norm = ', mc.norm())
        loss = 999999
    else:
        loss = -abs(lbm.rb.ang_vel[None].z * 1e3)
    return loss, prob