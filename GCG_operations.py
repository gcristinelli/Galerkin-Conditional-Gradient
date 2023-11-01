from dolfin import *
from mshr import *
import numpy as np
import time
from matplotlib import pyplot as pp


# ---------------------------------------------------------------------------------------------------------------------
def TV(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, cut_result):
    var1 = cut_result.vector()[intcells[:, 1]] - cut_result.vector()[intcells[:, 2]]
    TV = sum(intlengths * np.abs(var1)) + sum(boundlengths * np.abs(cut_result.vector()[boundfaces]))
    return TV


def clean_graph(mesh, vol_face_fn, bdy_length_fn, graph, dual, coeff, n):
    coeff = np.append(0, coeff)
    val = coeff[1] * dual.vector() * vol_face_fn.vector() + bdy_length_fn.vector()
    val_source = -np.maximum(0.0, val)
    val_sink = -np.maximum(0.0, -val)
    for i in range(2, n + 2):
        val = (coeff[i] - coeff[i - 1]) * dual.vector() * vol_face_fn.vector()
        val_source -= np.maximum(0.0, val)
        val_sink -= np.maximum(0.0, -val)
    graph.add_grid_tedges(np.arange(mesh.num_cells()), val_source, val_sink)


def _sparsify(U, K, c):
    ind = np.where(c <= 0.0000001)[0]
    U = np.delete(U, ind, 1)
    K = np.delete(K, ind, 1)
    c = np.delete(c, ind, 0)
    return U, K, c


def _linear_problem(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, graph, dual,
                    coeff, n, cut_result):
    coeff = np.append(0, coeff)
    val = (coeff[n + 1] - coeff[n]) * dual.vector() * vol_face_fn.vector() + float(n == 0) * bdy_length_fn.vector()
    graph.add_grid_tedges(np.arange(mesh.num_cells()), np.maximum(0.0, val), np.maximum(0.0, -val))
    energy = graph.maxflow()
    cut_result.vector()[:] = graph.get_grid_segments(np.arange(mesh.num_cells())).astype(float)
    integral = assemble(dual * cut_result * dx)
    per = TV(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, cut_result)
    return energy, integral, per


def _Dinkelbach(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, graph, dual, sign,
                alpha, flog, j, max_iterations=15, tolerance=1e-10):
    V = FunctionSpace(mesh, 'DG', 0)  # PWC
    coeff = np.zeros(max_iterations)
    per = np.zeros(max_iterations)
    integral_value = np.zeros(max_iterations)
    energy = np.zeros(max_iterations)

    # initializing coefficients
    flog.write("  Dinkelbach:\n")
    coeff[0] = sign * (1 / alpha)
    n = 0

    # ---initial cut
    cut_result = Function(V)
    prep_time = time.time()
    energy[n], integral_value[n], per[n] = _linear_problem(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells,
                                                           boundlengths, boundfaces, graph, dual, coeff, n, cut_result)
    flog.write("    The initial cut took %.2f seconds, has value - %.10f\n" % (
        (time.time() - prep_time), per[n] + coeff[n] * integral_value[n]))

    # if it is a zero cut, stop
    if per[n] < tolerance:
        flog.write("    Zero initial cut with coefficient %s \n" % (coeff[0]))
        ext = 0
        clean_time = time.time()
        clean_graph(mesh, vol_face_fn, bdy_length_fn, graph, dual, coeff, n)
        flog.write("    Cleaning took -%s\n" % (time.time() - clean_time))
        return cut_result, ext, per[n]
    else:
        flog.write("    The perimeter of the initial cut is %s\n" % per[n])
        n += 1

    # ---Following cuts (main loop)
    while (n == 1) or (
            1 < n < max_iterations and np.abs(per[n - 1] + coeff[n - 1] * integral_value[n - 1]) > tolerance):
        prev_time = time.time()
        coeff[n] = (-per[n - 1] / integral_value[n - 1])
        oldcut = cut_result
        cut_result = Function(V)
        energy[n], integral_value[n], per[n] = _linear_problem(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells,
                                                               boundlengths, boundfaces, graph, dual, coeff, n,
                                                               cut_result)
        flog.write("    Cut number %s has lambda equal to %s, took %.2f seconds, has value %.10f\n" % (
            n, coeff[n], (time.time() - prev_time), per[n] + coeff[n] * integral_value[n]))
        if per[n] < tolerance:
            flog.write("    Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n]))
            print("Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n]))
            ext = -1 / coeff[n]
            clean_time = time.time()
            clean_graph(mesh, vol_face_fn, bdy_length_fn, graph, dual, coeff, n)
            flog.write("    Cleaning took -%s\n" % (time.time() - clean_time))
            return oldcut, ext, per[n - 1]
        flog.write("    Its perimeter is %s\n" % per[n])
        # increment iterations
        n += 1

    flog.write("    Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n - 1]))
    print("Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n - 1]))
    ext = -1 / coeff[n - 1]
    clean_time = time.time()
    clean_graph(mesh, vol_face_fn, bdy_length_fn, graph, dual, coeff, n - 1)
    flog.write("    Cleaning took %.2f seconds\n" % (time.time() - clean_time))
    return cut_result, ext, per[n - 1]


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


def _export_ply(mesh, intcells, fun, j, index, threshold, rd, flog):
    start_time = time.time()
    mesh.init(2, 0)
    f2v = mesh.topology()(2, 0)
    var = np.abs(fun.vector()[intcells[:, 2]] - fun.vector()[intcells[:, 1]])
    var_index = np.column_stack((intcells[:, 0], var))
    non_zero_var = var_index[var_index[:, 1] > threshold]
    normalized_var = (255 * (non_zero_var[:, 1] - np.min(non_zero_var[:, 1])) / (np.max(non_zero_var[:, 1]))).astype(
        int)
    n = len(non_zero_var[:, 0])
    triangles = np.empty((0, 3), dtype=int)
    header = "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement " \
             "face {}\nproperty list uchar int vertex_indices\nproperty uchar red \nproperty uchar green \nproperty " \
             "uchar blue \n end_header\n"
    if index == 0:
        f = open(rd + '/input.ply', "w")
        f.write(header.format(3 * n, n))
    elif index == 1:
        f = open(rd + '/cut_%s.ply' % j, "w")
        f.write(header.format(3 * n, n))
    else:
        f = open(rd + '/U_%s.ply' % j, "w")
        flog.write("  The maximum variation of U%s between cells is %.6e \n" % (j, np.max(non_zero_var)))
        f.write(header.format(3 * n, n))

    triangles = np.arange(3 * n, dtype=int)
    vertices_per_face = 3 * np.ones((n, 1), dtype=int)
    triangles = np.concatenate((vertices_per_face, triangles.reshape((-1, 3))), axis=1)

    for face_index in range(n):
        vertices = f2v(int(non_zero_var[face_index, 0]))
        for vertex in vertices:
            f.write(" ".join(str(coord) for coord in mesh.coordinates()[vertex]) + "\n")

    for face_index in range(n):
        f.write(" ".join(str(ind) for ind in triangles[face_index, :]) + " " + " ".join(
            str(normalized_var[face_index]) for ind in range(3)) + "\n")

    f.close()
    flog.write("  Creating the ply file took %.2f seconds \n" % (time.time() - start_time))


def plot_result(mesh, int_cells, flog, rd, f, n, j, d):
    if d == 2:
        ax = plot(f)
        pp.colorbar(ax, shrink=0.55, format='%01.3f')
        if n == 0: pp.savefig(rd + '/input_Yo.png', bbox_inches='tight', dpi=600)
        if n == 1: pp.savefig(rd + '/cut_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 2: pp.savefig(rd + '/U_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 3: pp.savefig(rd + '/P_%s.png' % j, bbox_inches='tight', dpi=600)
        pp.close()
    elif n != 3:
        _export_ply(mesh, int_cells, f, j, n, 1 / 255, rd, flog)

# ---------------------------------------------------------------------------------------------------------------------
