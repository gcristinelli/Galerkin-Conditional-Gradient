from dolfin import *
import numpy as np
import time
from matplotlib import pyplot as pp


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


def _export_ply(msh, int_cells, fun, j, index, threshold, rd, flog):
    start_time = time.time()
    msh.init(2, 0)
    f2v = msh.topology()(2, 0)
    var = np.abs(fun.vector()[int_cells[:, 2]] - fun.vector()[int_cells[:, 1]])
    var_index = np.column_stack((int_cells[:, 0], var))
    non_zero_var = var_index[var_index[:, 1] > threshold]
    if len(non_zero_var) == 0:
        return
    normalized_var = (255 * (non_zero_var[:, 1] - np.min(non_zero_var[:, 1])) / (np.max(non_zero_var[:, 1]))).astype(
        int)
    n = len(non_zero_var[:, 0])
    header = "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement " \
             "face {}\nproperty list uchar int vertex_indices\nproperty uchar red \nproperty uchar green \nproperty " \
             "uchar blue \n end_header\n"
    if index == 1:
        f = open(rd + '/cut_%s.ply' % j, "w")
        f.write(header.format(3 * n, n))
    elif index == 2:
        f = open(rd + '/U_%s.ply' % j, "w")
        flog.write("  The maximum variation of U%s between cells is %.6e \n" % (j, np.max(non_zero_var)))
        f.write(header.format(3 * n, n))

    triangles = np.arange(3 * n, dtype=int)
    vertices_per_face = 3 * np.ones((n, 1), dtype=int)
    triangles = np.concatenate((vertices_per_face, triangles.reshape((-1, 3))), axis=1)

    for face_index in range(n):
        vert = f2v(int(non_zero_var[face_index, 0]))
        for vertex in vert:
            f.write(" ".join(str(coord) for coord in msh.coordinates()[vertex]) + "\n")

    for face_index in range(n):
        f.write(" ".join(str(ind) for ind in triangles[face_index, :]) + " " + " ".join(
            str(normalized_var[face_index]) for ind in range(3)) + "\n")

    f.close()
    flog.write("  Creating the ply file took %.2f seconds \n" % (time.time() - start_time))


def plot_convergence(rd, opt):
    pp.plot(range(len(np.trim_zeros(opt))), np.log10(np.trim_zeros(opt)))
    pp.xlabel("iteration")
    pp.ylabel("convergence indicator")
    pp.savefig(rd + '/indicator.png', dpi=300)
    pp.close()


def plot_energy(rd, energy):
    pp.plot(range(len(np.trim_zeros(energy))), np.trim_zeros(energy))
    pp.xlabel("iteration")
    pp.ylabel("energy")
    # pp.ylim([0, 0.001])
    pp.savefig(rd + '/energy.png', dpi=300)
    pp.close()
