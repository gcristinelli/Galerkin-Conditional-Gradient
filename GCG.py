
__author__ = "Giacomo Cristinelli, José A. Iglesias and Daniel Walter"
__date__ = "November 1st, 2023"

import argparse
import configparser
import time
from datetime import datetime

import maxflow
import numpy as np
import os
import scipy.sparse as cp
from dolfin import *
from matplotlib import pyplot as pp

from GCG_operations import _sparsify, TV
from mesh import _create_mesh
from insertion import _Dinkelbach
from plots import plot_result, plot_convergence, plot_energy
from SSN import _SSN


# ----------------------------------------------------------------------------------------------------------------------
def _main_():
    # ---CREATING DIRECTORY AND LOG FILE ------------------------------------------------------------------------------
    # FEniCS log level
    set_log_level(30)

    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")
    rd = os.path.join(os.path.dirname(__file__), './results/' + dt_string)
    if not os.path.isdir(rd):
        os.makedirs(rd)
    flog = open(rd + '/log.txt', "w")

    # ---TIMER
    start_time = time.time()

    # A1.---STORING INFO IN THE LOG FILE
    with open('setup.conf', 'r') as conf_file:
        # Read the contents of the conf file
        conf_content = conf_file.read()

    # Open a text file for writing
    flog.write(conf_content + '\n')

    # ---GEOMETRY LOOP ------------------------------------------------------------------------------------------------

    flog.write("Geometry loop:\n")
    print("Starting geometry loop...\n")

    # B1.---MESH GENERATION AND FUNCTION SPACES
    mesh = _create_mesh(Random_mesh, d, lx1, lx2, ly1, ly2, lz1, lz2, Nx, Ny, Nz, N, N2, rd)

    V = FunctionSpace(mesh, 'DG', 0)  # PWC
    VL = FunctionSpace(mesh, 'CG', 1)  # PWL
    mesh.init()
    mesh.init(d - 1, d)
    e2f = mesh.topology()(d - 1, d)
    vol_face_fn = Function(V)
    bdy_length_fn = Function(V)
    bdy_length = np.empty(0)
    bdy_faces = np.empty(0)
    facet_size = np.empty(0)

    if d == 2:
        flog.write("  Made a mesh with {} vertices, and {} faces \n".format(mesh.num_vertices(), mesh.num_faces()))
        plot(mesh, linewidth=0.25)
        pp.savefig(rd + '/0_mesh.png', bbox_inches='tight', dpi=300)
        pp.close()
    elif d == 3:
        flog.write(
            "  Made a mesh with {} vertices, {} faces, and {} {}-dimensional cells \n".format(mesh.num_vertices(),
                                                                                              mesh.num_faces(),
                                                                                              mesh.num_cells(), d))
    # dividing into boundary or internal facets
    facets_list = np.arange(mesh.num_facets())
    bdy_facets = np.array([facet for facet in facets_list if len(e2f(facet)) == 1], dtype=int)
    internal_facets = np.setdiff1d(facets_list, bdy_facets)

    # Defining dimension dependent size of facets
    if d == 2:
        facet_size = np.array([Edge(mesh, edge).length() for edge in facets_list])
    elif d == 3:
        facet_size = np.array([Face(mesh, face).area() for face in facets_list])

    int_cells = np.array([[facet, e2f(facet)[0], e2f(facet)[1]] for facet in internal_facets], dtype=int)
    int_lengths = facet_size[internal_facets]
    vol_face_fn.vector()[:] = [Cell(mesh, cell).volume() for cell in range(mesh.num_cells())]
    mid_cell = [Cell(mesh, cell).midpoint().array() for cell in range(0, mesh.num_cells())]

    # Process boundary info if needed
    if boundary:
        for facet in bdy_facets:
            bdy_length_fn.vector()[e2f(facet)[0]] += facet_size[facet]
            bdy_faces = np.append(bdy_faces, e2f(facet)[0])
            bdy_length = np.append(bdy_length, facet_size[facet])

    # defining integrating measures
    domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    dx = Measure("dx", domain=mesh, subdomain_data=domains)

    # B2.---GRAPH GENERATION, creating graph with (d-1)-facets areas/length as weights
    G = maxflow.GraphFloat()
    G.add_nodes(mesh.num_cells())
    G.add_edges(int_cells[:, 1], int_cells[:, 2], facet_size[int_cells[:, 0]], facet_size[int_cells[:, 0]])

    flog.write("  Constructed graph has {} nodes, and {} edges \n".format(G.get_node_count(), G.get_edge_count()))
    flog.write("  Making the mesh and graph took - %.2f seconds \n" % (time.time() - start_time))

    print("Making the mesh of {} vertices, {} {}-dimensional cells, and the graph took - {} seconds \n".format(
        mesh.num_vertices(), mesh.num_cells(), d, time.time() - start_time))

    # ---PDE AND OPERATORS CONSTRUCTION--------------------------------------------------------------------------------

    # C1.---PDE INFO
    g = Constant(0.0)
    bdr = DirichletBC(VL, g, DomainBoundary())
    v = TestFunction(VL)
    u = TrialFunction(VL)

    mass_form = v * u * dx
    M = assemble(mass_form)
    mat = as_backend_type(M).mat()
    ai, aj, av = mat.getValuesCSR()
    M = cp.csr_matrix((av, aj, ai))

    """# C2.---REFERENCE CONTROL AND OBSERVATIONS FOR CIRCLES
    Lap = 1 * inner(grad(u), grad(v)) * dx + 0.5 * v * u * dx
    U0 = interpolate(Constant(1.0), V)
    u0 = U0 * v * dx
    U0_arr = U0.vector().get_local()
    Yd = Function(VL)
    UD = Function(V)
    Y0 = Function(VL)
    shift = np.ones((mesh.num_cells(), d))

    if d == 2:
        shift = np.hstack([shift, np.zeros((mesh.num_cells(), 1))])

    UD.vector()[:] = 2 * (np.linalg.norm(mid_cell + 0.33 * shift, axis=1) < sqrt(0.1)) + (
            np.linalg.norm(mid_cell + (-0.33) * shift, axis=1) < sqrt(0.15)) - 1

    energy_D = alpha * TV(mesh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_length, bdy_faces, UD)

    flog.write("Energy of the toy control is %.8e\n" % energy_D)

    ud = UD * v * dx

    solve(Lap == ud, Yd, bdr, solver_parameters={'linear_solver': 'mumps'})
    solve(Lap == u0, Y0, bdr, solver_parameters={'linear_solver': 'mumps'})

    plot_result(mesh, int_cells, flog, rd, Yd, 0, 0, d)
    plot_result(mesh, int_cells, flog, rd, UD, 2, "o", d)"""

    # C2.---REFERENCE CONTROL AND OBSERVATIONS FOR CASTLE
    Lap = 1 * inner(grad(u), grad(v)) * dx
    U0 = interpolate(Constant(1.0), V)
    u0 = U0 * v * dx
    if d == 2:
        Yd = interpolate(Expression('''x[0]>-0.5&&x[0]<0.5&&x[1]>-0.5&&x[1]<0.5? 1.00001:0.0''', degree=0), V)
    else:
        Yd = interpolate(
            Expression('''x[0]>-0.5&&x[0]<0.5&&x[1]>-0.5&&x[1]<0.5&&x[2]>-0.5&&x[2]<0.5? 1.00001:0.0''', degree=0), V)

    plot_result(mesh, int_cells, flog, rd, Yd, 0, 0, d)

    Yd = interpolate(Yd, VL)

    Y0 = Function(VL)
    solve(Lap == u0, Y0, bdr, solver_parameters={'linear_solver': 'mumps'})

    # C3.---COEFFICIENTS, EXTREMALS, STATES
    measurements = Yd.vector().get_local()
    U0_arr = U0.vector().get_local()
    Y0_arr = Y0.vector().get_local()
    Km = np.reshape(Y0.vector().get_local(), (-1, 1))
    Kl = np.empty(shape=[len(Km), 0])
    mean = np.array([1])
    coefficients = np.empty(0)
    Ul = np.empty(shape=[len(U0_arr), 0])

    # --GENERALIZED CONDITIONAL GRADIENT-------------------------------------------------------------------------------

    # D1.---WARM UP ITERATION
    coefficients, mean, adjoint, optval, misfit = _SSN(Kl, Km, coefficients, mean, measurements, alpha, M)
    j = 0
    Uk = Function(V)
    prev_Uk = Function(V)
    Yk = Function(VL)
    Vk = Function(VL)
    Pk = Function(VL)
    Uk.vector()[:] = mean.flatten() * U0_arr + Ul @ coefficients
    Yk.vector()[:] = mean.flatten() * Y0_arr + Kl @ coefficients

    opt = np.zeros(max_iterations + 1)
    energy = np.zeros(max_iterations + 1)
    rel_change = np.zeros(max_iterations + 1)
    data = []
    total_time = 0

    energy[0] = 0.5 * assemble((Y0 - Yd) ** 2 * dx)
    export = [0, total_time, energy[0], optval/alpha, 0]
    data.append(export)

    # D2.---MAIN LOOP
    while (j == 0) or (j <= max_iterations and opt[j - 1] > tolerance):
        start_iteration_time = time.time()
        flog.write("Iteration {} of Conditional gradient: \n".format(j))
        print("Starting iteration %s of GCG\n" % j)

        prev_Uk.assign(Uk)

        # Solve adjoint equation
        rhpk = (Yk - Yd) * v * dx
        prev_time = time.time()
        solve(Lap == rhpk, Pk, bdr, solver_parameters={'linear_solver': 'mumps'})
        flog.write("  Adjoint PDE solve took %.2f seconds \n" % (time.time() - prev_time))

        # making sure pk has zero average
        Pkp = interpolate(Pk, V)
        intergral_pk = assemble(Pkp * dx)
        volume_domain = assemble(interpolate(Constant(1.0), V) * dx)
        Pkv = np.subtract(Pkp.vector(), intergral_pk / volume_domain)
        Pkp.vector()[:] = Pkv[:]
        flog.write("  Average of pk was %.6e \n" % assemble(Pkp * dx))

        if boundary:
            vm, extm, perm = _Dinkelbach(mesh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_length,
                                         bdy_faces, G, Pkp, 1, alpha, flog, j, tolerance=tolerance)
            vp, extp, perp = _Dinkelbach(mesh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_length,
                                         bdy_faces, G, Pkp, -1, alpha, flog, j, tolerance=tolerance)
            flog.write("  The new extremal coefficients are {}, {} \n".format(extm, extp))
            if np.abs(extp) > np.abs(extm):
                rhvk = -(vp / perp) * v * dx
                Ul = np.append(Ul, np.reshape(-vp.vector().get_local() / perp, (-1, 1)), axis=1)
            else:
                rhvk = (vm / perm) * v * dx
                Ul = np.append(Ul, np.reshape(vm.vector().get_local() / perm, (-1, 1)), axis=1)
        else:
            vm, extm, perm = _Dinkelbach(mesh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_length,
                                         bdy_faces, G, Pkp, 1, alpha, flog, j, tolerance=tolerance)
            flog.write("  The new extremal coefficient is {} \n".format(extm))
            if perm < tolerance:
                raise Exception("Zero cut at iteration %s" % j)
            plot_result(mesh, int_cells, flog, rd, vm, 1, j, d)
            rhvk = (vm / perm) * v * dx
            Ul = np.append(Ul, np.reshape(vm.vector().get_local() / perm, (-1, 1)), axis=1)

        prev_time = time.time()
        solve(Lap == rhvk, Vk, bdr, solver_parameters={'linear_solver': 'mumps'})
        flog.write("  new state PDE solve took %.2f seconds\n" % (time.time() - prev_time))

        Kl = np.append(Kl, np.reshape(Vk.vector().get_local(), (-1, 1)), axis=1)
        coefficients = np.append(coefficients, np.array([1]))

        # Updating coefficients with a semismooth Newton method
        coefficients, mean, adjoint, optval, misfit = _SSN(Kl, Km, coefficients, mean, measurements, alpha, M)

        opt[j] = ((np.abs(extm) - alpha) / alpha) * optval
        Ul, Kl, coefficients = _sparsify(Ul, Kl, coefficients)
        
        # Updating control and state
        Uk.vector()[:] = mean.flatten() * U0_arr + Ul @ coefficients
        Yk.vector()[:] = mean.flatten() * Y0_arr + Kl @ coefficients

        energy[j] = misfit + alpha * TV(mesh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_length,
                                        bdy_faces, Uk)
        rel_change[j] = assemble(abs(Uk - prev_Uk) * dx)/assemble(abs(Uk) * dx)

        plot_convergence(rd, opt)
        plot_energy(rd, energy)

        flog.write("  Current surrogate energy value is %.6e, with convergence indicator, %.6e \n" % (optval, opt[j]))
        flog.write("  Current actual energy value is %.6e \n" % (energy[j]))
        print("Step %s of GCG finished with energy value %.6e and convergence indicator %.6e \n" % (j, optval, opt[j]))

        plot_result(mesh, int_cells, flog, rd, Uk, 2, j, d)
        plot_result(mesh, int_cells, flog, rd, Pkp, 3, j, d)

        total_time += time.time() - start_iteration_time
        export = [j+1, total_time, energy[j], opt[j], rel_change[j]]
        data.append(export)
        j += 1

    data_array = np.array(data, dtype=float)
    energy_change = data_array[:, 2] - data_array[-1, 2]
    data_array = np.hstack((data_array, energy_change[:, np.newaxis]))
    fmt = ['%d', '%.2f', '%.6e', '%.6e', '%.6e', '%.6e']
    header = 'iteration, time (seconds), energy, indicator, L1-relative change, energy difference'
    np.savetxt(rd + '/output.csv', data_array, delimiter=',', fmt=fmt, header=header)

    print("Algorithm converged in %s steps, the total time was %.2f seconds" % (j - 1, time.time() - start_time))
    flog.write("The total time was %.2f seconds" % (time.time() - start_time))
    flog.close()


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Step 1: Create an instance of ArgumentParser
    parser = argparse.ArgumentParser()

    # Step 2: Add the --config argument
    dir = os.path.join(os.path.dirname(__file__), 'setup.conf')
    parser.add_argument('--config', type=str, help=dir, default='./setup.conf')

    # Step 3: Parse the command-line arguments
    args = parser.parse_args()

    # Step 4: Read the configuration file
    config = configparser.ConfigParser()
    config.read(args.config)

    # Step 5: Access values from the configuration file
    Random_mesh = eval(config.get('Mesh', 'Random_mesh', fallback='True'))
    Nx, Ny, Nz = eval(config.get('Mesh', 'Size', fallback='[200,200,200]'))
    N = eval(config.get('Mesh', 'N2D', fallback='100'))
    N2 = eval(config.get('Mesh', 'N3D', fallback='50'))
    d = eval(config.get('Mesh', 'dimension', fallback='2'))

    alpha = eval(config.get('Regularizer', 'Parameter', fallback='0.0001'))
    boundary = eval(config.get('Regularizer', 'Boundary', fallback='False'))

    lx1, lx2 = eval(config.get('Control', 'Domain_x', fallback='[-1,1]'))
    ly1, ly2 = eval(config.get('Control', 'Domain_y', fallback='[-1,1]'))
    lz1, lz2 = eval(config.get('Control', 'Domain_z', fallback='[-1,1]'))

    max_iterations = eval(config.get('GCG', 'max_iterations', fallback='120'))
    tolerance = eval(config.get('GCG', 'tolerance', fallback='1e-10'))

    _main_()
