from rigid import *
import time
from pyevtk.hl import gridToVTK
# ti.init(arch=ti.gpu)
float_type = ti.f32
mat3 = ti.types.matrix(3, 3, float_type)
vec3 = ti.types.vector(3, float_type)


@ti.data_oriented
class LBM3D:
    def __init__(self, nx, ny, nz, nu, info):
        self.nu = nu
        self.tau = 3 * nu + .5
        self.nx, self.ny, self.nz = nx, ny, nz
        self.fx, self.fy, self.fz = 0.0, 0.0, 0.0
        self.cs = 1 / ti.sqrt(3.)
        self.max_v = ti.field(float_type, shape=())
        self.C_mat = [[0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1],
             [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1],
             [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1]]
        self.H2 = ti.Matrix.field(3, 3, float_type, shape=27)
        self.H3_xxy = ti.field(float_type, shape=27)
        self.H3_xyy = ti.field(float_type, shape=27)
        self.H3_xxz = ti.field(float_type, shape=27)
        self.H3_xzz = ti.field(float_type, shape=27)
        self.H3_yzz = ti.field(float_type, shape=27)
        self.H3_yyz = ti.field(float_type, shape=27)
        self.H3_xyz = ti.field(float_type, shape=27)
        self.f = ti.Vector.field(27, float_type, shape=(nx, ny, nz))
        self.F = ti.Vector.field(27, float_type, shape=(nx, ny, nz))
        self.rho = ti.field(float_type, shape=(nx, ny, nz))
        self.v = ti.Vector.field(3, float_type, shape=(nx, ny, nz))
        self.vor = ti.Vector.field(3, float_type, shape=(nx, ny, nz))
        self.vor_norm = ti.field(float_type, shape=(nx, ny, nz))
        self.e = ti.Vector.field(3, ti.i32, shape=27)
        self.e_f = ti.Vector.field(3, float_type, shape=27)
        self.S = ti.Matrix.field(3, 3, float_type, shape=(nx, ny, nz))
        self.w = ti.field(float_type, shape=27)
        self.force = ti.Vector.field(3, float_type, shape=(nx, ny, nz))
        self.vel_rb = ti.Vector.field(3, float_type, shape=())
        self.external_force = ti.Vector.field(3, float_type, shape=())
        self.LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25]
        self.x = np.linspace(0, nx - 1, nx)
        self.y = np.linspace(0, ny - 1, ny)
        self.z = np.linspace(0, nz - 1, nz)
        self.bd_con = [0] * 6
        self.vel_bd = ti.Vector.field(3, float_type, shape=6)
        self.rb = rigid_body(nx, ny, nz, info)
        self.t = [0] * 5

        for i in range(27):
            e_i = ti.Vector([self.C_mat[0][i], self.C_mat[1][i], self.C_mat[2][i]], float_type)
            e_ii = ti.Vector([self.C_mat[0][i], self.C_mat[1][i], self.C_mat[2][i]], ti.i32)
            self.e_f[i] = e_i
            self.e[i] = e_ii
            if i == 0:
                self.w[i] = 8. / 27.
            elif 1 <= i <= 6:
                self.w[i] = 2. / 27.
            elif 7 <= i <= 18:
                self.w[i] = 1. / 54.
            elif 19 <= i <= 26:
                self.w[i] = 1. / 216.
            self.H2[i] = e_i.outer_product(e_i) - 1. / 3. * mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.H3_xxy[i] = e_i[0] * e_i[0] * e_i[1] - 1. / 3. * e_i[1]
            self.H3_xyy[i] = e_i[0] * e_i[1] * e_i[1] - 1. / 3. * e_i[0]
            self.H3_xxz[i] = e_i[0] * e_i[0] * e_i[2] - 1. / 3. * e_i[2]
            self.H3_xzz[i] = e_i[0] * e_i[2] * e_i[2] - 1. / 3. * e_i[0]
            self.H3_yzz[i] = e_i[1] * e_i[2] * e_i[2] - 1. / 3. * e_i[1]
            self.H3_yyz[i] = e_i[1] * e_i[1] * e_i[2] - 1. / 3. * e_i[2]
            self.H3_xyz[i] = e_i[0] * e_i[1] * e_i[2]

    def init_simulation(self):
        self.init_fields()

    @ti.kernel
    def init_fields(self):
        self.vel_rb[None] = vec3(0)
        self.external_force[None] = vec3(0)
        for i in ti.grouped(self.rho):
            self.rho[i] = 1.0
            self.v[i] = ti.Vector([0, 0, 0])
            self.force[i] = self.external_force[None]
            self.S[i] = mat3(0)
        for i in ti.grouped(self.rho):
            rho = self.rho[i]
            v = self.v[i]
            S = self.S[i]
            for s in ti.static(range(27)):
                self.f[i][s] = self.reconstruct_F_local(v, rho, S, s)
                self.F[i][s] = self.reconstruct_F_local(v, rho, S, s)
        for i in ti.grouped(self.rho):
            if 0 == i.x or 0 == i.y or 0 == i.z or self.nx - 1 == i.x or self.ny - 1 == i.y or self.nz - 1 == i.z:
                self.v[i] = -self.rb.lin_vel[None]


    @ti.kernel
    def collision(self):
        tau = self.tau
        for i in ti.grouped(self.rho):
            v = self.v[i]
            rho = self.rho[i]
            f = self.force[i]
            self.v[i] += (f / 2) / rho
            S_new = S = self.S[i]
            Sxx = S[0, 0]
            Syy = S[1, 1]
            Szz = S[2, 2]
            Sxy = S[0, 1]
            Syz = S[1, 2]
            Sxz = S[0, 2]
            S_new[1, 0] = S_new[0, 1] = ((1. - 1. / tau) * Sxy + 1. / tau * v.x * v.y +
                                         (2. * tau - 1.) / (2. * tau * rho) * (f.x * v.y + f.y * v.x))
            S_new[1, 2] = S_new[2, 1] = ((1. - 1. / tau) * Syz + 1. / tau * v.y * v.z +
                                         (2. * tau - 1.) / (2. * tau * rho) * (f.z * v.y + f.y * v.z))
            S_new[2, 0] = S_new[0, 2] = ((1. - 1. / tau) * Sxz + 1. / tau * v.x * v.z +
                                         (2. * tau - 1.) / (2. * tau * rho) * (f.x * v.z + f.z * v.x))
            S_new[0, 0] = ((tau - 1.) / (3. * tau) * (2. * Sxx - Syy - Szz) +
                           (v.x ** 2. + v.y ** 2. + v.z ** 2.) / 3 +
                           (2. * v.x ** 2. - v.y ** 2. - v.z ** 2.) / (3. * tau) +
                           f.x * v.x / rho +
                           (tau - 1.) / (3. * tau * rho) * (2. * f.x * v.x - f.y * v.y - f.z * v.z))
            S_new[1, 1] = ((tau - 1.) / (3. * tau) * (2. * Syy - Sxx - Szz) +
                           1. / 3. * (v.x ** 2. + v.y ** 2. + v.z ** 2.) +
                           (2. * v.y ** 2. - v.x ** 2. - v.z ** 2.) / (3. * tau) +
                           f.y * v.y / rho +
                           (tau - 1.) / (3. * tau * rho) * (2. * f.y * v.y - f.x * v.x - f.z * v.z))
            S_new[2, 2] = ((tau - 1.) / (3. * tau) * (2. * Szz - Syy - Sxx) +
                           1. / 3. * (v.x ** 2. + v.y ** 2. + v.z ** 2.) +
                           (2. * v.z ** 2. - v.y ** 2. - v.x ** 2.) / (3. * tau) +
                           f.z * v.z / rho +
                           (tau - 1.) / (3. * tau * rho) * (2. * f.z * v.z - f.y * v.y - f.x * v.x))
            self.S[i] = S_new
            # self.v[i] += (1. - 1./(2 * tau)) * f / rho
            # S = self.S[i]
            # self.S[i] = (1. - 1./tau) * S + 1./tau * v.outer_product(v) + (1. - 1./(2. * tau)) / rho * (f.outer_product(v) + v.outer_product(f))

    @ti.func
    def periodic_index(self, i):
        i_out = i
        if i[0] < 0:
            i_out[0] = self.nx - 1
        if i[0] > self.nx - 1:
            i_out[0] = 0
        if i[1] < 0:
            i_out[1] = self.ny - 1
        if i[1] > self.ny - 1:
            i_out[1] = 0
        if i[2] < 0:
            i_out[2] = self.nz - 1
        if i[2] > self.nz - 1:
            i_out[2] = 0
        return i_out

    @ti.kernel
    def My_streaming(self):
        for i in ti.grouped(self.rho):
            for s in ti.static(range(27)):
                fp = 0.
                frac = self.rb.cell_marker[i][self.LR[s]]
                if frac < 0:
                    ip = i - self.e[s]
                    if ((ip[0] >= 0) and (ip[0] < self.nx) and (ip[1] >= 0) and (ip[1] < self.ny) and (ip[2] >= 0) and (
                            ip[2] < self.nz)):
                        fp = self.f[ip][s]
                    else:
                        fp = self.feq(s, 1.0, -self.vel_rb[None])
                else:
                    # print(i,"ooo")
                    p = frac * self.e_f[self.LR[s]] + i
                    vp = self.rb.arbitrary_vel(p)
                    vx = self.v[i]
                    rhop = self.rho[i]
                    Sp = vp.outer_product(vp) + self.S[i] - vx.outer_product(vx)
                    fp = self.reconstruct_F_local(vp, rhop, Sp, s)
                self.F[i][s] = fp

    @ti.kernel
    def streaming(self):
        force = vec3(0)
        torque = vec3(0)
        for i in ti.grouped(self.rho):
            local_force = vec3(0)
            for s in ti.static(range(27)):
                fp = 0.
                frac = self.rb.cell_marker[i][self.LR[s]]
                if frac < 0:
                    ip = i - self.e[s]
                    if 0 <= ip.x < self.nx and 0 <= ip.y < self.ny and 0 <= ip.z < self.nz:
                        fp = self.f[ip][s]
                    else:
                        fp = self.feq(s, 1.0, -self.vel_rb[None])
                else:
                    p = frac * self.e_f[self.LR[s]] + i
                    vp = self.rb.arbitrary_vel(p)
                    vx = self.v[i]
                    rhop = self.rho[i]
                    Sp = vp.outer_product(vp) + self.S[i] - vx.outer_product(vx)
                    fp = self.reconstruct_F_local(vp, rhop, Sp, s)
                    local_force = local_force + self.f[i][self.LR[s]] * (self.e_f[self.LR[s]] - vp) - fp * (
                            self.e_f[s] - vp)
                self.F[i][s] = fp
            force += local_force
            torque += (i - self.rb.mass_center[None]).cross(local_force)
        self.rb.force[None] = force
        self.rb.torque[None] = torque

    @ti.kernel
    def My_force_from_fluid_to_object(self):
        force = vec3(0.)
        torque = vec3(0.)
        for i in ti.grouped(self.rho):
            local_force = vec3(0)
            for s in ti.static(range(27)):
                frac = self.rb.cell_marker[i][self.LR[s]]
                if frac > 0:
                    p = frac * self.e_f[self.LR[s]] + i
                    vp = self.rb.arbitrary_vel(p)
                    local_force += self.f[i][self.LR[s]] * (self.e_f[self.LR[s]] - vp) - self.F[i][s] * (
                                self.e_f[s] - vp)
                    # if i.x == 14 and i.y == 17 and i.z == 16:
                    #     print(self.f_temp[i][s], "ooo", s)
            # if abs(local_force.z) > 0:
            #     print(local_force, i)
            force += local_force
            torque += (i - self.rb.mass_center[None]).cross(local_force)
            # if(abs(local_force.z) > 0):
            #     print(force, i)
        self.rb.force[None] = force
        self.rb.torque[None] = torque


    @ti.func
    def feq(self, k, rho_local, u):
        eu = self.e_f[k].dot(u)
        uu = u.dot(u)
        feq = self.w[k] * rho_local * (1.0 +
                                       eu / (self.cs ** 2) +
                                       eu * eu / (2 * self.cs ** 4) -
                                       uu / (2 * self.cs ** 2))
        return feq

    @ti.func
    def reconstruct_F_local(self, v, rho, S, s):
        e_f = self.e_f[s]
        part1 = 1 + e_f.dot(v) / self.cs ** 2
        part2 = (self.H2[s] * S).sum() / (2 * self.cs ** 4)
        part3_0 = (self.H3_xxy[s] * (S[0, 0] * v.y + 2 * S[0, 1] * v.x - 2 * v.x * v.x * v.y))
        part3_1 = (self.H3_xyy[s] * (S[1, 1] * v.x + 2 * S[0, 1] * v.y - 2 * v.x * v.y * v.y))
        part3_2 = (self.H3_xxz[s] * (S[0, 0] * v.z + 2 * S[0, 2] * v.x - 2 * v.x * v.x * v.z))
        part3_3 = (self.H3_xzz[s] * (S[2, 2] * v.x + 2 * S[0, 2] * v.z - 2 * v.x * v.z * v.z))
        part3_4 = (self.H3_yzz[s] * (S[2, 2] * v.y + 2 * S[1, 2] * v.z - 2 * v.y * v.z * v.z))
        part3_5 = (self.H3_yyz[s] * (S[1, 1] * v.z + 2 * S[1, 2] * v.z - 2 * v.y * v.y * v.z))
        part3_6 = (self.H3_xyz[s] * (S[0, 2] * v.y + S[1, 2] * v.x + S[0, 1] * v.z - 2 * v.x * v.y * v.z))
        part3 = (part3_0 + part3_1 + part3_2 + part3_3 + part3_4 + part3_5 + part3_6) / (2 * self.cs ** 6)
        return rho * self.w[s] * (part1 + part2 + part3)
        # re = rho * self.w[s] * (1 + 3 * v.dot(e_f) + (self.H2[s] * S).sum() / (2 * self.cs ** 4))
        # return re

    @ti.kernel
    def construct_distribution(self):
        for i in ti.grouped(self.rho):
            rho = self.rho[i]
            v = self.v[i]
            S = self.S[i]
            for s in ti.static(range(27)):
                self.f[i][s] = self.reconstruct_F_local(v, rho, S, s)

    @ti.kernel
    def compute_moments(self):
        for i in ti.grouped(self.rho):
            rho = 0.
            S = mat3(0)
            v = vec3(0)
            for s in ti.static(range(27)):
                F = self.F[i][s]
                rho = rho + F
                v = v + self.e_f[s] * F
                S = S + self.H2[s] * F
            self.rho[i] = rho
            self.S[i] = S / rho
            self.v[i] = (v + self.force[i] / 2) / rho

    @ti.kernel
    def boundary_condition_F(self):
        if ti.static(self.bd_con[0] == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                for s in range(27):
                    self.F[0, j, k][s] = self.feq(s, 1, self.vel_bd[0])

        if ti.static(self.bd_con[1] == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                for s in range(27):
                    self.F[self.nx - 1, j, k][s] = self.feq(s, 1, self.vel_bd[1])

        if ti.static(self.bd_con[2] == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                for s in range(27):
                    self.F[i, 0, k][s] = self.feq(s, 1, self.vel_bd[2])

        if ti.static(self.bd_con[3] == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                for s in range(27):
                    self.F[i, self.ny - 1, k][s] = self.feq(s, 1, self.vel_bd[3])

        if ti.static(self.bd_con[4] == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                for s in range(27):
                    self.F[i, j, 0][s] = self.feq(s, 1, self.vel_bd[4])

        if ti.static(self.bd_con[5] == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                for s in range(27):
                    self.F[i, j, self.nz - 1][s] = self.feq(s, 1, self.vel_bd[5])

    @ti.kernel
    def boundary_condition_f(self):
        if ti.static(self.bd_con[0] == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                for s in range(27):
                    self.f[0, j, k][s] = self.feq(s, 1, self.vel_bd[0])

        if ti.static(self.bd_con[1] == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                for s in range(27):
                    self.f[self.nx - 1, j, k][s] = self.feq(s, 1, self.vel_bd[1])

        if ti.static(self.bd_con[2] == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                for s in range(27):
                    self.f[i, 0, k][s] = self.feq(s, 1, self.vel_bd[2])

        if ti.static(self.bd_con[3] == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                for s in range(27):
                    self.f[i, self.ny - 1, k][s] = self.feq(s, 1, self.vel_bd[3])

        if ti.static(self.bd_con[4] == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                for s in range(27):
                    self.f[i, j, 0][s] = self.feq(s, 1, self.vel_bd[4])

        if ti.static(self.bd_con[5] == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                for s in range(27):
                    self.f[i, j, self.nz - 1][s] = self.feq(s, 1, self.vel_bd[5])

    def set_vel_bd(self, vel, i):
        self.vel_bd[i] = ti.Vector(vel)
        self.bd_con[i] = 1

    @ti.kernel
    def compute_vorticity(self):
        for i in ti.grouped(self.rho):
            ix = i + ivec3(1, 0, 0)
            iy = i + ivec3(0, 1, 0)
            iz = i + ivec3(0, 0, 1)
            v_ix = vec3(0)
            v_iy = vec3(0)
            v_iz = vec3(0)
            v_temp = - self.vel_rb[None]
            if ix.x < self.nx:
                v_ix = self.v[ix]
            else:
                v_ix = v_temp
            if iy.y < self.ny:
                v_iy = self.v[iy]
            else:
                v_iy = v_temp
            if iz.z < self.nz:
                v_iz = self.v[iz]
            else:
                v_iz = v_temp
            w1 = (v_iy.z - self.v[i].z) - (v_iz.y - self.v[i].y)
            w2 = (v_iz.x - self.v[i].x) - (v_ix.z - self.v[i].z)
            w3 = (v_ix.y - self.v[i].y) - (v_iy.x - self.v[i].x)
            self.vor[i] = vec3(w1, w2, w3)
            self.vor_norm[i] = vec3(w1, w2, w3).norm()

    def export_VTK(self, n):
        print("output frame {}".format(n))
        self.compute_vorticity()
        self.rb.output("./bin/vis/mesh_{}.ply".format(n))
        self.rb.output_global("./bin/vis/gmesh_{}.ply".format(n))
        v = self.v.to_numpy()
        w = self.vor.to_numpy()
        gridToVTK(
            "./bin/vis/lbm_" + str(n),
            self.x,
            self.y,
            self.z,
            # cellData={},
            pointData={"rho": self.rho.to_numpy(),
                       "vorticity norm": self.vor_norm.to_numpy(),
                       "vorticity": (np.ascontiguousarray(w[:, :, :, 0]),
                                     np.ascontiguousarray(w[:, :, :, 1]),
                                     np.ascontiguousarray(w[:, :, :, 2])),
                       "velocity": (np.ascontiguousarray(self.v.to_numpy()[0:self.nx, 0:self.ny, 0:self.nz, 0]),
                                    np.ascontiguousarray(self.v.to_numpy()[0:self.nx, 0:self.ny, 0:self.nz, 1]),
                                    np.ascontiguousarray(self.v.to_numpy()[0:self.nx, 0:self.ny, 0:self.nz, 2]))
                       }

        )

    @ti.kernel
    def handle_non_intertial_frame(self):
        self.vel_rb[None] = self.rb.lin_vel[None]
        acc = self.rb.acc[None]
        for i in ti.grouped(self.rho):
            self.force[i] = self.external_force[None] - self.rho[i] * acc

    def step(self):
        # t0 = time.time()
        self.construct_distribution()
        # t1 = time.time()
        # self.streaming()
        self.My_streaming()
        self.My_force_from_fluid_to_object()
        # t2 = time.time()
        self.rb.step(1)
        self.handle_non_intertial_frame()
        # t3 = time.time()
        self.compute_moments()
        # t4 = time.time()
        self.collision()
        # t5 = time.time()
        # self.t[0] += t1 - t0
        # self.t[1] += t2 - t1
        # self.t[2] += t3 - t2
        # self.t[3] += t4 - t3
        # self.t[4] += t5 - t4
