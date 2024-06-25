from dolfin import *
from mshr import *
import numpy as np
import time

from GCG_operations import TV

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