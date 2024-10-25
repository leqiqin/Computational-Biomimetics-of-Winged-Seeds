import taichi as ti
import meshtaichi_patcher as Patcher
import numpy as np
import time

float_type = ti.f32
ivec3 = ti.math.ivec3
mat3 = ti.types.matrix(3, 3, float_type)
vec3 = ti.types.vector(3, float_type)

def rot(axis, a, inv=False):
    sin, cos = a[0], a[1]
    if inv:
        sin = -sin
    if axis == 2:
        return mat3([cos, -sin, 0], [sin, cos, 0], [0, 0, 1])
    if axis == 1:
        return mat3([cos, 0, sin], [0, 1, 0], [-sin, 0, cos])
    if axis == 0:
        return mat3([1, 0, 0], [0, cos, -sin], [0, sin, cos])


@ti.func
def mat_rot(axis, angle):
    sin = ti.sin(angle)
    cos = ti.cos(angle)
    R = mat3(0)
    if axis == 2:
        R = mat3([cos, -sin, 0], [sin, cos, 0], [0, 0, 1])
    if axis == 1:
        R = mat3([cos, 0, sin], [0, 1, 0], [-sin, 0, cos])
    if axis == 0:
        R = mat3([1, 0, 0], [0, cos, -sin], [0, sin, cos])
    return R


sin45 = ti.sqrt(2.) / 2.
cos45 = sin45
a45 = [sin45, cos45]
a135 = [sin45, -cos45]
a225 = [-sin45, -cos45]
a315 = [-sin45, cos45]
sin36 = 1. / ti.sqrt(3)
cos36 = ti.sqrt(2. / 3.)
a36 = [sin36, cos36]
rot_x_36 = rot(0, a36)
rot_x_36_inv = rot(0, a36, True)
trans_1 = rot(2, a45)
trans_1_inv = rot(2, a45, True)
trans_2 = rot(1, a45)
trans_2_inv = rot(1, a45, True)
trans_3 = rot(0, a45)
trans_3_inv = rot(0, a45, True)
trans_4 = rot(2, a135)
trans_4_inv = rot(2, a135, True)
trans_5 = rot(2, a225)
trans_5_inv = rot(2, a225, True)
trans_6 = rot(2, a315)
trans_6_inv = rot(2, a315, True)
trans_7 = rot_x_36 @ trans_1
trans_7_inv = trans_1_inv @ rot_x_36_inv
trans_8 = rot_x_36 @ trans_4
trans_8_inv = trans_4_inv @ rot_x_36_inv
trans_9 = rot_x_36 @ trans_5
trans_9_inv = trans_5_inv @ rot_x_36_inv
trans_10 = rot_x_36 @ trans_6
trans_10_inv = trans_6_inv @ rot_x_36_inv


class rigid_info:
    def __init__(self, mesh_name, mesh_mass, mesh_mass_center_global, seed_mass_center,
                 seed_mass_ratio, mesh_scale, mesh_translation, mesh_init_amom, mesh_init_vel, gravity, mesh_euler):
        self.mesh_name = mesh_name
        self.mesh_mass = mesh_mass
        self.mesh_mass_center_global = mesh_mass_center_global
        self.mesh_scale = mesh_scale
        self.mesh_translation = mesh_translation
        self.mesh_init_amom = mesh_init_amom
        self.mesh_init_vel = mesh_init_vel
        self.gravity = gravity
        self.mesh_euler = mesh_euler
        self.seed_mass_center = seed_mass_center
        self.seed_mass_ratio = seed_mass_ratio


@ti.data_oriented
class rigid_body:
    def __init__(self, _nx, _ny, _nz, info):
        self.mesh = Patcher.load_mesh(info.mesh_name, relations=["FV", "VF"])
        self.mesh.verts.place({'pos': vec3, 'radius': vec3, 'dual_area': float_type})
        self.mesh.faces.place({'verts_id': ivec3, 'area': float_type})
        self.pos_numpy = self.mesh.get_position_as_numpy()
        self.nx = _nx
        self.ny = _ny
        self.nz = _nz
        self.cell_marker = ti.Vector.field(27, shape=(self.nx, self.ny, self.nz), dtype=float_type)
        self.primitive_marker = ti.Vector.field(27, shape=(self.nx, self.ny, self.nz), dtype=ti.i32)
        self.inter_marker_tester = ti.Vector.field(3, shape=(self.nx, self.ny, self.nz), dtype=float_type)
        self.inter_test = []

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy())
        self.mass_center = ti.Vector.field(3, shape=(), dtype=float_type)
        self.seed_mass_center = ti.Vector.field(3, shape=(), dtype=float_type)
        self.seed_mass_ratio = 0.5
        self.mass = 0
        self.nverts = len(self.mesh.verts)
        self.nfaces = len(self.mesh.faces)
        self.grav_accr = vec3(0, 0, info.gravity)
        self.force = ti.Vector.field(3, shape=(), dtype=float_type)
        self.torque = ti.Vector.field(3, shape=(), dtype=float_type)
        self.ang_mom = ti.Vector.field(3, shape=(), dtype=float_type)
        self.lin_vel = ti.Vector.field(3, shape=(), dtype=float_type)
        self.ang_vel = ti.Vector.field(3, shape=(), dtype=float_type)
        self.frame = ti.Matrix.field(3, 3, shape=(), dtype=float_type)
        self.inertia_tensor = ti.Matrix.field(3, 3, shape=(), dtype=float_type)
        self.inertia_tensor_inv = ti.Matrix.field(3, 3, shape=(), dtype=float_type)
        self.acc = ti.Vector.field(3, shape=(), dtype=ti.f32)
        self.global_mass_center = ti.Vector.field(3, shape=(), dtype=ti.f32)
        self.load_face_adj_verts_id()

    def reinitialize_simulation(self, info, pos_torch):
        self.mesh.verts.pos.from_torch(pos_torch)
        self.seed_mass_center[None] = info.seed_mass_center
        self.seed_mass_ratio = info.seed_mass_ratio
        self.global_mass_center[None] = info.mesh_mass_center_global
        self.grav_accr = vec3(0, 0, info.gravity)
        self.mass = info.mesh_mass
        self.compute_mass_center()
        self.transform(info.mesh_scale, info.mesh_translation, info.mesh_euler)
        self.frame[None] = mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.lin_vel[None] = info.mesh_init_vel
        self.ang_mom[None] = info.mesh_init_amom
        self.force[None] = self.torque[None] = self.ang_vel[None] = self.acc[None] = vec3(0)
        self.compute_radius()
        self.compute_inertia_tensor()
        self.compute_intersections()

    @ti.kernel
    def compute_radius(self):
        for v in self.mesh.verts:
            v.radius = v.pos - self.mass_center[None]

    @ti.kernel
    def compute_mass_center(self):
        m_seed = self.seed_mass_ratio * self.mass
        m_rest = self.mass - m_seed

        c_rest = vec3(0.0)
        for v in self.mesh.verts:
            c_rest += v.pos
        self.mass_center[None] = m_rest * c_rest / self.nverts

        self.mass_center[None] += m_seed * self.seed_mass_center[None]

        self.mass_center[None] /= self.mass

    @ti.kernel
    def compute_inertia_tensor(self):
        # for v in self.mesh.verts:
        #     m = self.mass / self.nverts
        #     I_v = m * (v.radius.norm_sqr() * ti.Matrix.identity(float_type, 3) - v.radius.outer_product(v.radius))
        #     I += I_v
        m_seed = self.seed_mass_ratio * self.mass
        m_v = (1.0 - self.seed_mass_ratio) * self.mass / self.nverts

        I = mat3(0)

        for v in self.mesh.verts:
            I_v = m_v * (v.radius.norm_sqr() * ti.Matrix.identity(ti.f32, 3) - v.radius.outer_product(v.radius))
            I += I_v

        seed_radius = self.seed_mass_center[None] - self.mass_center[None]
        I += m_seed * (seed_radius.norm_sqr() * ti.Matrix.identity(ti.f32, 3) - seed_radius.outer_product(seed_radius))
        # sphere seed assumption
        # r = 0.5 * self.scale[0]
        r = 1.5
        Ix = 0.4 * m_seed * r * r
        I[0, 0] += Ix
        I[1, 1] += Ix
        I[2, 2] += Ix


        self.inertia_tensor[None] = I
        self.inertia_tensor_inv[None] = I.inverse()

    def step(self, dt):
        self.integrate(dt)
        self.reconstruct_verts_position()
        self.compute_intersections()

    @ti.kernel
    def integrate(self, dt: float_type):
        self.acc[None] = self.force[None] / self.mass + self.grav_accr
        self.lin_vel[None] += self.acc[None] * dt
        self.global_mass_center[None] += self.lin_vel[None] * dt
        self.ang_mom[None] += self.torque[None] * dt
        self.ang_vel[None] = self.frame[None] @ self.inertia_tensor_inv[None] @ self.frame[None].transpose() @ \
                             self.ang_mom[None]
        R = self.mat_rotation(self.ang_vel[None], dt)
        self.frame[None] = R @ self.frame[None]

    @ti.kernel
    def reconstruct_verts_position(self):
        for v in self.mesh.verts:
            v.pos = self.mass_center[None] + self.frame[None] @ v.radius

    @ti.func
    def face_intersection1(self, a, b, c, id):
        tri_min = ti.min(a, ti.min(b, c))
        tri_max = ti.max(a, ti.max(b, c))
        starts = ti.ceil(tri_min, ti.i32)

        j = starts.y
        while j < tri_max.y:
            k = starts.z
            while k < tri_max.z:
                result, alpha, beta, gamma = self.point_in_triangle(j, k, a.y, a.z, b.y, b.z, c.y, c.z)
                if result:
                    x_depth = a.x * alpha + b.x * beta + c.x * gamma
                    i = ti.floor(x_depth, ti.i32)
                    frac = x_depth - i
                    if 0 <= x_depth <= self.nx and 0 <= j <= self.ny and 0 <= k <= self.nz:
                        self.cell_marker[i, j, k][1] = frac
                        self.primitive_marker[i, j, k][1] = id
                        self.cell_marker[i + 1, j, k][2] = 1 - frac
                        self.primitive_marker[i + 1, j, k][2] = id
                k += 1
            j += 1

        k = starts.z
        while k < tri_max.z:
            i = starts.x
            while i < tri_max.x:
                result, alpha, beta, gamma = self.point_in_triangle(k, i, a.z, a.x, b.z, b.x, c.z, c.x)
                if result:
                    y_depth = a.y * alpha + b.y * beta + c.y * gamma
                    j = ti.floor(y_depth, ti.i32)
                    frac = y_depth - j
                    if 0 <= i <= self.nx and 0 <= y_depth <= self.ny and 0 <= k <= self.nz:
                        self.cell_marker[i, j, k][3] = frac
                        self.primitive_marker[i, j, k][3] = id
                        self.cell_marker[i, j + 1, k][4] = 1 - frac
                        self.primitive_marker[i, j + 1, k][4] = id
                i += 1
            k += 1

        i = starts.x
        while i < tri_max.x:
            j = starts.y
            while j < tri_max.y:
                result, alpha, beta, gamma = self.point_in_triangle(i, j, a.x, a.y, b.x, b.y, c.x, c.y)
                if result:
                    z_depth = a.z * alpha + b.z * beta + c.z * gamma
                    k = ti.floor(z_depth, ti.i32)
                    frac = z_depth - k
                    if 0 <= i <= self.nx and 0 <= j <= self.ny and 0 <= z_depth <= self.nz:
                        self.cell_marker[i, j, k][5] = frac
                        self.primitive_marker[i, j, k][5] = id
                        self.cell_marker[i, j, k + 1][6] = 1 - frac
                        self.primitive_marker[i, j, k + 1][6] = id
                j += 1
            i += 1

    @ti.func
    def face_intersection2(self, a, b, c, id):
        ta = trans_1 @ a
        tb = trans_1 @ b
        tc = trans_1 @ c
        tri_min = ti.min(ta, ti.min(tb, tc))
        tri_max = ti.max(ta, ti.max(tb, tc))
        starts_x = ti.ceil(tri_min.x / sin45, float_type) * sin45
        starts_y = ti.ceil(tri_min.y / sin45, float_type) * sin45
        starts_z = ti.ceil(tri_min.z, ti.i32)
        k = starts_z
        while k < tri_max.z:
            i = starts_x
            while i < tri_max.x:
                result, alpha, beta, gamma = self.point_in_triangle(k, i, ta.z, ta.x, tb.z, tb.x, tc.z, tc.x)
                if result:
                    y_depth = ta.y * alpha + tb.y * beta + tc.y * gamma
                    intp = trans_1_inv @ vec3(i, y_depth, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.floor(intp, ti.i32)
                        intpi.z = k
                        frac = (intp - intpi).norm() / ti.sqrt(2)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][7] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][7] = id
                        self.cell_marker[intpi.x + 1, intpi.y + 1, intpi.z][8] = 1 - frac
                        self.primitive_marker[intpi.x + 1, intpi.y + 1, intpi.z][8] = id
                i += sin45
            k += 1

        j = starts_y
        while j < tri_max.y:
            k = starts_z
            while k < tri_max.z:
                result, alpha, beta, gamma = self.point_in_triangle(j, k, ta.y, ta.z, tb.y, tb.z, tc.y, tc.z)
                if result:
                    x_depth = ta.x * alpha + tb.x * beta + tc.x * gamma
                    intp = trans_1_inv @ vec3(x_depth, j, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.floor(intp, ti.i32)
                        intpi.z = k
                        intpi.x += 1
                        frac = (intp - intpi).norm() / ti.sqrt(2)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][14] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][14] = id
                        self.cell_marker[intpi.x - 1, intpi.y + 1, intpi.z][13] = 1 - frac
                        self.primitive_marker[intpi.x - 1, intpi.y + 1, intpi.z][13] = id
                k += 1
            j += sin45

    @ti.func
    def face_intersection3(self, a, b, c, id):
        ta = trans_2 @ a
        tb = trans_2 @ b
        tc = trans_2 @ c
        tri_min = ti.min(ta, ti.min(tb, tc))
        tri_max = ti.max(ta, ti.max(tb, tc))
        starts_x = ti.ceil(tri_min.x / sin45, float_type) * sin45
        starts_y = ti.ceil(tri_min.y, ti.i32)
        starts_z = ti.ceil(tri_min.z / sin45, float_type) * sin45
        j = starts_y
        while j < tri_max.y:
            k = starts_z
            while k < tri_max.z:
                result, alpha, beta, gamma = self.point_in_triangle(j, k, ta.y, ta.z, tb.y, tb.z, tc.y, tc.z)
                if result:
                    x_depth = ta.x * alpha + tb.x * beta + tc.x * gamma
                    intp = trans_2_inv @ vec3(x_depth, j, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.floor(intp, ti.i32)
                        intpi.y = j
                        frac = (intp - intpi).norm() / ti.sqrt(2)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][9] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][9] = id
                        self.cell_marker[intpi.x + 1, intpi.y, intpi.z + 1][10] = 1 - frac
                        self.primitive_marker[intpi.x + 1, intpi.y, intpi.z + 1][10] = id
                k += sin45
            j += 1

        i = starts_x
        while i < tri_max.x:
            j = starts_y
            while j < tri_max.y:
                result, alpha, beta, gamma = self.point_in_triangle(i, j, ta.x, ta.y, tb.x, tb.y, tc.x, tc.y)
                if result:
                    z_depth = ta.z * alpha + tb.z * beta + tc.z * gamma
                    intp = trans_2_inv @ vec3(i, j, z_depth)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.floor(intp, ti.i32)
                        intpi.y = j
                        intpi.x += 1
                        frac = (intp - intpi).norm() / ti.sqrt(2)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][16] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][16] = id
                        self.cell_marker[intpi.x - 1, intpi.y, intpi.z + 1][15] = 1 - frac
                        self.primitive_marker[intpi.x - 1, intpi.y, intpi.z + 1][15] = id
                j += 1
            i += sin45

    @ti.func
    def face_intersection4(self, a, b, c, id):
        ta = trans_3 @ a
        tb = trans_3 @ b
        tc = trans_3 @ c
        tri_min = ti.min(ta, ti.min(tb, tc))
        tri_max = ti.max(ta, ti.max(tb, tc))
        starts_x = ti.ceil(tri_min.x, ti.i32)
        starts_y = ti.ceil(tri_min.y / sin45, float_type) * sin45
        starts_z = ti.ceil(tri_min.z / sin45, float_type) * sin45
        i = starts_x
        while i < tri_max.x:
            j = starts_y
            while j < tri_max.y:
                result, alpha, beta, gamma = self.point_in_triangle(i, j, ta.x, ta.y, tb.x, tb.y, tc.x, tc.y)
                if result:
                    z_depth = ta.z * alpha + tb.z * beta + tc.z * gamma
                    intp = trans_3_inv @ vec3(i, j, z_depth)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.floor(intp, ti.i32)
                        intpi.x = i
                        frac = (intp - intpi).norm() / ti.sqrt(2)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][11] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][11] = id
                        self.cell_marker[intpi.x, intpi.y + 1, intpi.z + 1][12] = 1 - frac
                        self.primitive_marker[intpi.x, intpi.y + 1, intpi.z + 1][12] = id
                j += sin45
            i += 1

        k = starts_z
        while k < tri_max.z:
            i = starts_x
            while i < tri_max.x:
                result, alpha, beta, gamma = self.point_in_triangle(k, i, ta.z, ta.x, tb.z, tb.x, tc.z, tc.x)
                if result:
                    y_depth = ta.y * alpha + tb.y * beta + tc.y * gamma
                    intp = trans_3_inv @ vec3(i, y_depth, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.floor(intp, ti.i32)
                        intpi.x = i
                        intpi.z += 1
                        frac = (intp - intpi).norm() / ti.sqrt(2)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][17] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][17] = id
                        self.cell_marker[intpi.x, intpi.y + 1, intpi.z - 1][18] = 1 - frac
                        self.primitive_marker[intpi.x, intpi.y + 1, intpi.z - 1][18] = id
                i += 1
            k += sin45

    @ti.func
    def face_intersection5(self, a, b, c, id):
        ta = trans_7 @ a
        tb = trans_7 @ b
        tc = trans_7 @ c
        tri_min = ti.min(ta, ti.min(tb, tc))
        tri_max = ti.max(ta, ti.max(tb, tc))
        starts_x = ti.ceil(tri_min.x / sin45, ti.i32)
        starts_z_even = ti.ceil(tri_min.z / cos36, float_type) * cos36
        starts_z_odd = ti.ceil((tri_min.z - 1 / ti.sqrt(6.)) / cos36, float_type) * cos36 + (1 / (ti.sqrt(6.)))
        i = starts_x
        while i * sin45 < tri_max.x:
            k = starts_z_odd
            if i % 2 == 0:
                k = starts_z_even
            while k < tri_max.z:
                result, alpha, beta, gamma = self.point_in_triangle(i * sin45, k, ta.x, ta.z, tb.x, tb.z, tc.x, tc.z)
                if result:
                    y_depth = ta.y * alpha + tb.y * beta + tc.y * gamma
                    intp = trans_7_inv @ vec3(i * sin45, y_depth, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.ceil(intp, ti.i32)
                        intpi.y -= 1
                        intpi.x -= 1
                        frac = (intp - intpi).norm() / ti.sqrt(3)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][21] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][21] = id
                        self.cell_marker[intpi.x + 1, intpi.y + 1, intpi.z - 1][22] = 1 - frac
                        self.primitive_marker[intpi.x + 1, intpi.y + 1, intpi.z - 1][22] = id
                k += cos36
            i += 1

    @ti.func
    def face_intersection6(self, a, b, c, id):
        ta = trans_8 @ a
        tb = trans_8 @ b
        tc = trans_8 @ c
        tri_min = ti.min(ta, ti.min(tb, tc))
        tri_max = ti.max(ta, ti.max(tb, tc))
        starts_x = ti.ceil(tri_min.x / sin45, ti.i32)
        starts_z_even = ti.ceil(tri_min.z / cos36, float_type) * cos36
        starts_z_odd = ti.ceil((tri_min.z - 1 / ti.sqrt(6.)) / cos36, float_type) * cos36 + (1 / (ti.sqrt(6.)))
        i = starts_x
        while i * sin45 < tri_max.x:
            k = starts_z_odd
            if i % 2 == 0:
                k = starts_z_even
            while k < tri_max.z:
                result, alpha, beta, gamma = self.point_in_triangle(i * sin45, k, ta.x, ta.z, tb.x, tb.z, tc.x, tc.z)
                if result:
                    y_depth = ta.y * alpha + tb.y * beta + tc.y * gamma
                    intp = trans_8_inv @ vec3(i * sin45, y_depth, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.ceil(intp, ti.i32)
                        intpi.x -= 1
                        frac = (intp - intpi).norm() / ti.sqrt(3)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][26] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][26] = id
                        self.cell_marker[intpi.x + 1, intpi.y - 1, intpi.z - 1][25] = 1 - frac
                        self.primitive_marker[intpi.x + 1, intpi.y - 1, intpi.z - 1][25] = id
                k += cos36
            i += 1

    @ti.func
    def face_intersection7(self, a, b, c, id):
        ta = trans_9 @ a
        tb = trans_9 @ b
        tc = trans_9 @ c
        tri_min = ti.min(ta, ti.min(tb, tc))
        tri_max = ti.max(ta, ti.max(tb, tc))
        starts_x = ti.ceil(tri_min.x / sin45, ti.i32)
        starts_z_even = ti.ceil(tri_min.z / cos36, float_type) * cos36
        starts_z_odd = ti.ceil((tri_min.z - 1 / ti.sqrt(6.)) / cos36, float_type) * cos36 + (1 / (ti.sqrt(6.)))
        i = starts_x
        while i * sin45 < tri_max.x:
            k = starts_z_odd
            if i % 2 == 0:
                k = starts_z_even
            while k < tri_max.z:
                result, alpha, beta, gamma = self.point_in_triangle(i * sin45, k, ta.x, ta.z, tb.x, tb.z, tc.x, tc.z)
                if result:
                    y_depth = ta.y * alpha + tb.y * beta + tc.y * gamma
                    intp = trans_9_inv @ vec3(i * sin45, y_depth, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.ceil(intp, ti.i32)
                        frac = (intp - intpi).norm() / ti.sqrt(3)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][20] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][20] = id
                        self.cell_marker[intpi.x - 1, intpi.y - 1, intpi.z - 1][19] = 1 - frac
                        self.primitive_marker[intpi.x - 1, intpi.y - 1, intpi.z - 1][19] = id
                k += cos36
            i += 1

    @ti.func
    def face_intersection8(self, a, b, c, id):
        ta = trans_10 @ a
        tb = trans_10 @ b
        tc = trans_10 @ c
        tri_min = ti.min(ta, ti.min(tb, tc))
        tri_max = ti.max(ta, ti.max(tb, tc))
        starts_x = ti.ceil(tri_min.x / sin45, ti.i32)
        starts_z_even = ti.ceil(tri_min.z / cos36, float_type) * cos36
        starts_z_odd = ti.ceil((tri_min.z - 1 / ti.sqrt(6.)) / cos36, float_type) * cos36 + (1 / (ti.sqrt(6.)))
        i = starts_x
        while i * sin45 < tri_max.x:
            k = starts_z_odd
            if i % 2 == 0:
                k = starts_z_even
            while k < tri_max.z:
                result, alpha, beta, gamma = self.point_in_triangle(i * sin45, k, ta.x, ta.z, tb.x, tb.z, tc.x, tc.z)
                if result:
                    y_depth = ta.y * alpha + tb.y * beta + tc.y * gamma
                    intp = trans_10_inv @ vec3(i * sin45, y_depth, k)
                    if 0 <= intp.x <= self.nx and 0 <= intp.y <= self.ny and 0 <= intp.z <= self.nz:
                        intpi = ti.ceil(intp, ti.i32)
                        intpi.y -= 1
                        frac = (intp - intpi).norm() / ti.sqrt(3)
                        self.cell_marker[intpi.x, intpi.y, intpi.z][24] = frac
                        self.primitive_marker[intpi.x, intpi.y, intpi.z][24] = id
                        self.cell_marker[intpi.x - 1, intpi.y + 1, intpi.z - 1][23] = 1 - frac
                        self.primitive_marker[intpi.x - 1, intpi.y + 1, intpi.z - 1][23] = id
                k += cos36
            i += 1

    @ti.kernel
    def compute_intersections(self):
        self.cell_marker.fill(-1)
        self.primitive_marker.fill(-1)
        for f in self.mesh.faces:
            a = f.verts[0].pos
            b = f.verts[1].pos
            c = f.verts[2].pos
            id = f.id
            self.face_intersection1(a, b, c, id)
            self.face_intersection2(a, b, c, id)
            self.face_intersection3(a, b, c, id)
            self.face_intersection4(a, b, c, id)
            self.face_intersection5(a, b, c, id)
            self.face_intersection6(a, b, c, id)
            self.face_intersection7(a, b, c, id)
            self.face_intersection8(a, b, c, id)

    @ti.func
    def point_in_triangle(self, p0, p1, a0, a1, b0, b1, c0, c1):
        alpha = -(p0 - b0) * (c1 - b1) + (p1 - b1) * (c0 - b0)
        alpha /= -(a0 - b0) * (c1 - b1) + (a1 - b1) * (c0 - b0)
        beta = -(p0 - c0) * (a1 - c1) + (p1 - c1) * (a0 - c0)
        beta /= -(b0 - c0) * (a1 - c1) + (b1 - c1) * (a0 - c0)
        gamma = 1.0 - alpha - beta
        result = 0.0 < alpha < 1.0 and 0.0 < beta < 1.0 and gamma > 0.0
        return result, alpha, beta, gamma

    @ti.kernel
    def load_face_adj_verts_id(self):
        for face in self.mesh.faces:
            face.verts_id = ivec3(face.verts[0].id, face.verts[1].id, face.verts[2].id)

    @ti.func
    def mat_cross(self, a):
        return mat3([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    @ti.func
    def arbitrary_vel(self, p: vec3):
        r = p - self.mass_center[None]
        # v = self.ang_vel[None].cross(r) + self.lin_vel[None]
        v = self.ang_vel[None].cross(r)
        return v

    @ti.func
    def mat_rotation(self, omega, dt):
        theta = omega.norm() * dt
        I = mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v = omega.normalized()
        cos = ti.cos(theta)
        R = cos * I + (1. - cos) * v.outer_product(v) + ti.sin(theta) * self.mat_cross(v)
        if theta < 1e-10:
            R = I
        return R

    @ti.kernel
    def transform(self, scale: vec3, translation: vec3, euler: vec3):
        R = mat_rot(0, euler.x) @ mat_rot(1, euler.y) @ mat_rot(2, euler.z)
        self.mass_center[None] = R @ (self.mass_center[None] * scale)
        trans = translation - self.mass_center[None]
        self.mass_center[None] = translation
        self.seed_mass_center[None] = R @ (self.seed_mass_center[None] * scale) + trans
        # self.mass_center[None] = R @ (self.mass_center[None] * scale) + translation
        # self.seed_mass_center[None] = R @ (self.seed_mass_center[None] * scale) + translation
        for v in self.mesh.verts:
            # v.pos = R @ (v.pos * scale) + translation
            v.pos = R @ (v.pos * scale) + trans

    @ti.kernel
    def compute_areas(self):
        self.total_area[None] = 0
        for f in self.mesh.faces:
            v0 = f.verts[0].pos
            v1 = f.verts[1].pos
            v2 = f.verts[2].pos
            e1 = v1 - v0
            e2 = v2 - v0
            f.area = (e1.cross(e2)).norm() / 2
            self.total_area[None] += f.area
        for v in self.mesh.verts:
            area = 0.
            for f in v.faces:
                area += f.area
            v.dual_area = area / 3

    def output(self, output_name):
        x_np = self.mesh.verts.pos.to_numpy()
        f_np = self.mesh.faces.verts_id.to_numpy().flatten()
        writer = ti.tools.PLYWriter(num_vertices=self.nverts, num_faces=self.nfaces)
        writer.add_vertex_pos(x_np[:, 0], x_np[:, 1], x_np[:, 2])
        writer.add_faces(f_np)
        writer.export_ascii(output_name)

    def output_global(self, output_name):
        gmc = self.global_mass_center[None]
        x_np = self.mesh.verts.pos.to_numpy()
        f_np = self.mesh.faces.verts_id.to_numpy().flatten()
        writer = ti.tools.PLYWriter(num_vertices=self.nverts, num_faces=self.nfaces)
        writer.add_vertex_pos(x_np[:, 0] + gmc.x, x_np[:, 1] + gmc.y, x_np[:, 2]+gmc.z)
        writer.add_faces(f_np)
        writer.export_ascii(output_name)

    def output_houdini(self, frame, output_name):
        x_np = self.mesh.verts.pos.to_numpy()
        f_np = self.mesh.faces.verts_id.to_numpy().flatten()
        writer = ti.tools.PLYWriter(num_vertices=self.nverts, num_faces=self.nfaces)
        writer.add_vertex_pos(x_np[:, 0], x_np[:, 1], x_np[:, 2])
        writer.add_faces(f_np)
        writer.export_frame(frame, output_name)