from dolfin import *
from mshr import *
import numpy as np
import time
from GCG_operations import TV


def _linear_problem(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, graph, dual,
                    coeff, n, cut_result):
    graph_copy = graph.copy()
    val = coeff * dual.vector() * vol_face_fn.vector() + bdy_length_fn.vector()
    graph_copy.add_grid_tedges(np.arange(mesh.num_cells()), np.maximum(0.0, val), np.maximum(0.0, -val))
    energy = graph_copy.maxflow()
    cut_result.vector()[:] = graph_copy.get_grid_segments(np.arange(mesh.num_cells())).astype(float)
    integral = assemble(dual * cut_result * dx)
    per = TV(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, cut_result)
    return energy, integral, per

def _Dinkelbach(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, graph, dual, sign,
                alpha, flog, j, max_iterations=15, tolerance=1e-10):
    V = FunctionSpace(mesh, 'DG', 0)  # PWC
    per = np.zeros(max_iterations)
    integral_value = np.zeros(max_iterations)
    energy = np.zeros(max_iterations)

    # initializing coefficients
    flog.write("  Dinkelbach:\n")
    coeff = sign * (1 / alpha)
    n = 0

    # ---initial cut
    cut_result = Function(V)
    prep_time = time.time()
    energy[n], integral_value[n], per[n] = _linear_problem(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells,
                                                           boundlengths, boundfaces, graph, dual, coeff, n, cut_result)
    flog.write("    The initial cut took %.2f seconds, has value - %.10f\n" % (
        (time.time() - prep_time), per[n] + coeff * integral_value[n]))

    # if it is a zero cut, stop
    if per[n] < tolerance:
        flog.write("    Zero initial cut with coefficient %s \n" % (coeff))
        ext = 0
        return cut_result, ext, per[n], n
    else:
        flog.write("    The perimeter of the initial cut is %s\n" % per[n])
        n += 1

    # ---Following cuts (main loop)
    while (n == 1) or (
            1 < n < max_iterations and np.abs(per[n - 1] + coeff * integral_value[n - 1]) > tolerance):
        prev_time = time.time()
        coeff = (-per[n - 1] / integral_value[n - 1])
        oldcut = cut_result
        cut_result = Function(V)
        energy[n], integral_value[n], per[n] = _linear_problem(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells,
                                                               boundlengths, boundfaces, graph, dual, coeff, n,
                                                               cut_result)
        flog.write("    Cut number %s has lambda equal to %s, took %.2f seconds, has value %.10f\n" % (
            n, coeff, (time.time() - prev_time), per[n] + coeff * integral_value[n]))
        if per[n] < tolerance:
            flog.write("    Dinkelbach stops after %s steps with lambda %.6e \n" % (n+1, coeff))
            print("Dinkelbach stops after %s steps with lambda %.6e \n" % (n+1, coeff))
            ext = -1 / coeff
            return oldcut, ext, per[n - 1], n + 1
        flog.write("    Its perimeter is %s\n" % per[n])
        # increment iterations
        n += 1

    flog.write("    Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff))
    print("Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff))
    ext = -1 / coeff
    return cut_result, ext, per[n - 1], n