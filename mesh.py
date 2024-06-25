from dolfin import *
from mshr import *
import numpy as np
import time

def _create_mesh(Random_mesh, d, lx1, lx2, ly1, ly2, lz1, lz2, Nx, Ny, Nz, N, N2, rd):
    if Random_mesh and d == 2:
        mesh = _symmetric_rectangle_mesh(lx2, ly2, N)
    elif Random_mesh and d == 3:
        domain = Box(Point(lx1, ly1, lz1), Point(lx2, ly2, lz2))
        mesh = generate_mesh(domain, N2)
        dolfin.cpp.io.XDMFFile(MPI.COMM_WORLD, rd + '/mesh.xdmf').write(mesh)
    elif not Random_mesh and d == 2:
        mesh = RectangleMesh(Point(lx1, ly1), Point(lx2, ly2), Nx, Ny, 'crossed')
    else:
        mesh = BoxMesh(Point(lx1, ly1, lz1), Point(lx2, ly2, lz2), Nx, Ny, Nz)
    return mesh

def _symmetric_rectangle_mesh(x1, y1, Nm):
    # first symmetrize in the x direction, then symmetrize result in y direction
    domain = Rectangle(Point(0.0, 0.0), Point(x1, y1))
    mesh = generate_mesh(domain, Nm)
    mesh.init()
    f2v = mesh.topology()(2, 0)

    start_time = time.time()

    mesh_sym = Mesh()
    editor = MeshEditor()
    editor.open(mesh_sym, 'triangle', 2, 2)

    nv = mesh.num_vertices()
    nf = mesh.num_cells()
    cntv = nv
    kv = 0
    kf = 0

    # First count how many vertices will be needed
    for jv in range(nv):
        vertex_obj = Vertex(mesh, jv)
        pnt_orig = vertex_obj.point()
        if not near(pnt_orig.x(), 0.0):
            cntv += 1

    editor.init_vertices(cntv)
    editor.init_cells(2 * mesh.num_cells())

    for jv in range(nv):
        vertex_obj = Vertex(mesh, jv)
        pnt_orig = vertex_obj.point()
        pnt = Point(pnt_orig.x(), pnt_orig.y())
        editor.add_vertex(kv, pnt)
        kv += 1
    for jf in range(nf):
        pnts = f2v(jf)
        editor.add_cell(kf, pnts.tolist())
        kf += 1

    kvf = kv
    nb = 0
    shifts = [nv] * nv
    for jv in range(nv):
        vertex_obj = Vertex(mesh, jv)
        pnt_orig = vertex_obj.point()
        # only add vertices not in the reflection boundary
        if not near(pnt_orig.x(), 0.0):
            pnt = Point((-1.0) * pnt_orig.x(), pnt_orig.y())
            editor.add_vertex(kv, pnt)
            shifts[jv] = kvf - nb
            kv += 1
        else:
            nb += 1
            shifts[jv] = 0
    for jf in range(nf):
        pnts = f2v(jf)
        vertex_obj = Vertex(mesh, pnts[0])
        pnt0 = vertex_obj.point()
        vertex_obj = Vertex(mesh, pnts[1])
        pnt1 = vertex_obj.point()
        vertex_obj = Vertex(mesh, pnts[2])
        pnt2 = vertex_obj.point()
        # if a vertex is in the reflection boundary, use original
        pnts = pnts + [shifts[pnts[0]], shifts[pnts[1]], shifts[pnts[2]]]
        editor.add_cell(kf, pnts.tolist())
        kf += 1
    editor.close()

    mesh = Mesh(mesh_sym)
    mesh.init()
    f2v = mesh.topology()(2, 0)
    mesh_sym = Mesh()
    editor = MeshEditor()
    editor.open(mesh_sym, 'triangle', 2, 2)
    nv = mesh.num_vertices()
    nf = mesh.num_cells()
    cntv = nv
    kv = 0
    kf = 0

    # First count how many vertices will be needed
    for jv in range(nv):
        vertex_obj = Vertex(mesh, jv)
        pnt_orig = vertex_obj.point()
        if not near(pnt_orig.y(), 0.0):
            cntv += 1

    editor.init_vertices(cntv)
    editor.init_cells(2 * mesh.num_cells())

    for jv in range(nv):
        vertex_obj = Vertex(mesh, jv)
        pnt_orig = vertex_obj.point()
        pnt = Point(pnt_orig.x(), pnt_orig.y())
        editor.add_vertex(kv, pnt)
        kv += 1
    for jf in range(nf):
        pnts = f2v(jf)
        editor.add_cell(kf, pnts.tolist())
        kf += 1

    kvf = kv
    nb = 0
    shifts = [nv] * nv
    for jv in range(nv):
        vertex_obj = Vertex(mesh, jv)
        pnt_orig = vertex_obj.point()
        # only add vertices not in the reflection boundary
        if not near(pnt_orig.y(), 0.0):
            pnt = Point(pnt_orig.x(), (-1.0) * pnt_orig.y())
            editor.add_vertex(kv, pnt)
            shifts[jv] = kvf - nb
            kv += 1
        else:
            nb += 1
            shifts[jv] = 0
    for jf in range(nf):
        pnts = f2v(jf)
        vertex_obj = Vertex(mesh, pnts[0])
        pnt0 = vertex_obj.point()
        vertex_obj = Vertex(mesh, pnts[1])
        pnt1 = vertex_obj.point()
        vertex_obj = Vertex(mesh, pnts[2])
        pnt2 = vertex_obj.point()
        # if a vertex is in the reflection boundary, use original
        pnts = pnts + [shifts[pnts[0]], shifts[pnts[1]], shifts[pnts[2]]]
        editor.add_cell(kf, pnts.tolist())
        kf += 1
    editor.close()
    return mesh_sym
