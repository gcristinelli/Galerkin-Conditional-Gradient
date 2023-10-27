
__author__ = "Giacomo Cristinelli, José A. Iglesias and Daniel Walter"
__date__ = "October 27th, 2023"

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
from mshr import *

from GCG_operations import _sparsify, _Dinkelbach, _symmetric_rectangle_mesh, _export_ply, TV
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
    flog.write("Variables:\n")

    if Random_mesh and d == 2:
        flog.write("  Random mesh in a rectangle of size {}x{}, with indicator {}\n".format(lx2 - lx1, ly2 - ly1, N))
    elif Random_mesh and d == 3:
        flog.write(
            "  Random mesh in a cube of size {}x{}x{}, with indicator {}\n".format(lx2 - lx1, ly2 - ly1, lz2 - lz1, N2))
    elif not Random_mesh and d == 2:
        flog.write("  Standard crossed mesh with {}x{} squares in a rectangle of size {}x{}\n".format(Nx, Ny, lx2 - lx1,
                                                                                                      ly2 - ly1))
    elif not Random_mesh and d == 3:
        flog.write("  Regular Box mesh with {}x{}x{} cubes in a cube of size {}x{}x{}\n".format(Nx, Ny, Nz, lx2 - lx1,
                                                                                                ly2 - ly1, lz2 - lz1))
    else:
        raise Exception("Not supported dimension. Choose d=2,3")

    flog.write("  The dimension is {}\n".format(d))
    flog.write("  The regularizer parameter is {}\n".format(alpha))
    flog.write("  TV with boundary: {}.\n".format(boundary))
    flog.write("  GCG has tolerance {}, and {} max iterations.\n".format(tolerance, max_iterations))

    # ---GEOMETRY LOOP ------------------------------------------------------------------------------------------------

    flog.write("Geometry loop:\n")
    print("Starting geometry loop...\n")

    # B1.---MESH GENERATION AND FUNCTION SPACES
    if Random_mesh and d == 2:
        mesh = _symmetric_rectangle_mesh(lx2, ly2, N)
    elif Random_mesh and d == 3:
        domain = Box(Point(lx1, ly1, lz1), Point(lx2, ly2, lz2))
        mesh = generate_mesh(domain, N2)
    elif not Random_mesh and d == 2:
        mesh = RectangleMesh(Point(lx1, ly1), Point(lx2, ly2), Nx, Ny, 'crossed')
    else:
        mesh = BoxMesh(Point(lx1, ly1, lz1), Point(lx2, ly2, lz2), Nx, Ny, Nz)

    V = FunctionSpace(mesh, 'DG', 0)  # PWC
    VL = FunctionSpace(mesh, 'CG', 1)  # PWL
    mesh.init()
    mesh.init(d - 1, d)
    e2f = mesh.topology()(d - 1, d)
    vol_face_fn = Function(V)
    bdy_length_fn = Function(V)
    int_lengths = np.empty(0)
    bdy_length = np.empty(0)
    int_cells = np.empty(shape=[0, 2])
    bdy_faces = np.empty(0)
    internal_facets = np.empty(0)
    facet_size = np.empty(0)
    bdy_facets = np.empty(0)

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

    # B2.---GRAPH GENERATION
    G = maxflow.GraphFloat()
    G.add_nodes(mesh.num_cells())

    # Defining dimension dependent objects
    if d == 2:
        edges_list = np.arange(mesh.num_edges())
        facet_size = np.array([Edge(mesh, edge).length() for edge in edges_list])
        internal_facets = np.array([edge for edge in edges_list if len(e2f(edge)) > 1], dtype=int)
        bdy_facets = np.setdiff1d(edges_list, internal_facets)
    elif d == 3:
        faces_list = np.arange(mesh.num_faces())
        facet_size = np.array([Face(mesh, face).area() for face in faces_list])
        internal_facets = np.array([face for face in faces_list if len(e2f(face)) > 1], dtype=int)
        bdy_facets = np.setdiff1d(faces_list, internal_facets)

    # Creating graph with (d-1)-facets areas/length as weights
    for facet in internal_facets:
        adj_cells = e2f(facet)
        size = facet_size[facet]
        G.add_edge(adj_cells[0], adj_cells[1], size, size)
        int_cells = np.append(int_cells, np.reshape(adj_cells, (1, -1)), axis=0)
        int_lengths = np.append(int_lengths, size)

    int_cells = np.concatenate((internal_facets.reshape((-1, 1)), int_cells), axis=1)
    # The first column gives the index of the facet, while the other two give the indices of the adjacent cells
    vol_face_fn.vector()[:] = [Cell(mesh, cell).volume() for cell in range(mesh.num_cells())]
    mid_cell = [Cell(mesh, cell).midpoint().array() for cell in range(0, mesh.num_cells())]

    # Process boundary info if needed
    if boundary:
        for facet in bdy_facets:
            cell = e2f(facet)[0]
            size = facet_size[facet]
            bdy_length_fn.vector()[cell] += size
            bdy_faces = np.append(bdy_faces, cell)
            bdy_length = np.append(bdy_length, size)

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
    measurements = Yd.vector().get_local()

    if d == 2:
        ax = plot(project(Yd, V))
        pp.colorbar(ax, shrink=0.55, format='%01.3f')
        pp.savefig(rd + '/input_Yd.png', bbox_inches='tight', dpi=600)
        pp.close()
        ax = plot(UD)
        pp.colorbar(ax, shrink=0.55, format='%01.3f')
        pp.savefig(rd + '/input_Ud.png', bbox_inches='tight', dpi=600)
        pp.close()
    else:
        _export_ply(mesh, int_cells, project(Yd, V), 0, True, 1e-10, rd, flog)
        _export_ply(mesh, int_cells, UD, "d", False, 1e-10, rd, flog)"""

    # C2.---REFERENCE CONTROL AND OBSERVATIONS FOR CASTLE
    Lap = 1 * inner(grad(u), grad(v)) * dx
    U0 = interpolate(Constant(1.0), V)
    u0 = U0 * v * dx
    U0_arr = U0.vector().get_local()
    if d == 2:
        Yd = interpolate(Expression('''x[0]>-0.5&&x[0]<0.5&&x[1]>-0.5&&x[1]<0.5? 1.00001:0.0''', degree=0), V)
    else:
        Yd = interpolate(
            Expression('''x[0]>-0.5&&x[0]<0.5&&x[1]>-0.5&&x[1]<0.5&&x[2]>-0.5&&x[2]<0.5? 1.00001:0.0''', degree=0), V)

    if d == 2:
        ax = plot(Yd, vmin=0.0, vmax=1.00001)
        pp.colorbar(ax, shrink=0.55, format='%01.3f')
        pp.savefig(rd + '/input_Yd.png', bbox_inches='tight', dpi=600)
        pp.close()
    else:
        _export_ply(mesh, int_cells, Yd, 0, True, 1e-10, rd, flog)

    Yd = interpolate(Yd, VL)

    Y0 = Function(VL)
    solve(Lap == u0, Y0, bdr, solver_parameters={'linear_solver': 'mumps'})
    measurements = Yd.vector().get_local()

    # C3.---COEFFICIENTS, EXTREMALS, STATES
    Km = np.reshape(Y0.vector().get_local(), (-1, 1))
    Kl = np.empty(shape=[len(Km), 0])
    mean = np.array([1])
    coefficients = np.empty(0)
    Ul = np.empty(shape=[len(U0_arr), 0])

    # --GENERALIZED CONDITIONAL GRADIENT-------------------------------------------------------------------------------

    # D1.---WARM UP ITERATION
    coefficients, mean, adjoint, optval, misfit = _SSN(Kl, Km, coefficients, mean, measurements, alpha, M)
    flog.write("Warm up iteration:\n")
    flog.write("  The mean is {} \n".format(mean))
    flog.write("  The coefficients are {} \n".format(coefficients))
    flog.write("  The adjoint is {}\n".format(adjoint))
    flog.write("  Current value is %.6e \n" % optval)

    j = 0
    Uk = Function(V)
    prev_Uk = Function(V)
    Uk.vector()[:] = mean.flatten() * U0_arr + Ul @ coefficients
    opt = np.zeros(max_iterations + 1)
    energy = np.zeros(max_iterations + 1)
    rel_change = np.zeros(max_iterations + 1)
    data = []
    total_time = 0

    # D2.---MAIN LOOP
    while (j == 0) or (j <= max_iterations and opt[j - 1] > tolerance):
        start_iteration_time = time.time()
        flog.write("Iteration {} of Conditional gradient: \n".format(j))
        print("Starting iteration %s of GCG\n" % j)
        # Solve state and adjoint equation
        rhk = Uk * v * dx
        Yk = Function(VL)
        Vk = Function(VL)
        Vkc = Function(VL)
        prev_time = time.time()
        prev_Uk = Uk
        solve(Lap == rhk, Yk, bdr, solver_parameters={'linear_solver': 'mumps'})
        flog.write("  First PDE solve took %.2f seconds \n" % (time.time() - prev_time))
        Pk = Function(VL)
        rhpk = (Yk - Yd) * v * dx
        prev_time = time.time()
        solve(Lap == rhpk, Pk, bdr, solver_parameters={'linear_solver': 'mumps'})
        flog.write("  Second PDE solve took %.2f seconds \n" % (time.time() - prev_time))

        # making sure pk has zero average
        Pkp = project(Pk, V)
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
            rhvk = (vm / perm) * v * dx
            Ul = np.append(Ul, np.reshape(vm.vector().get_local() / perm, (-1, 1)), axis=1)

        prev_time = time.time()
        solve(Lap == rhvk, Vk, bdr, solver_parameters={'linear_solver': 'mumps'})
        flog.write("  Last PDE solve took %.2f seconds\n" % (time.time() - prev_time))
        Kl = np.append(Kl, np.reshape(Vk.vector().get_local(), (-1, 1)), axis=1)
        coefficients = np.append(coefficients, np.array([1]))
        coefficients, mean, adjoint, optval, misfit = _SSN(Kl, Km, coefficients, mean, measurements, alpha, M)
        opt[j] = ((np.abs(extm) - alpha) / alpha) * optval
        Ul, Kl, coefficients = _sparsify(Ul, Kl, coefficients)
        Uk = Function(V)
        Uk.vector()[:] = mean.flatten() * U0_arr + Ul @ coefficients
        energy[j] = misfit + alpha * TV(mesh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_length,
                                        bdy_faces, Uk)
        rel_change[j] = assemble(abs(Uk - prev_Uk) * dx)/assemble(abs(Uk) * dx)

        pp.plot(range(len(np.trim_zeros(opt))), np.log10(np.trim_zeros(opt)))
        pp.xlabel("iteration")
        pp.ylabel("convergence indicator")
        pp.savefig(rd + '/indicator.png', dpi=300)
        pp.close()
        pp.plot(range(len(np.trim_zeros(energy))), np.trim_zeros(energy))
        pp.xlabel("iteration")
        pp.ylabel("energy")
        # pp.ylim([0, 0.001])
        pp.savefig(rd + '/energy.png', dpi=300)
        pp.close()
        flog.write("  Current surrogate energy value is %.6e, with convergence indicator, %.6e \n" % (optval, opt[j]))
        flog.write("  Current actual energy value is %.6e \n" % (energy[j]))
        print("Step %s of GCG finished with energy value %.6e and convergence indicator %.6e \n" % (j, optval, opt[j]))

        if d == 2:
            ax = plot(Uk)
            pp.colorbar(ax, shrink=0.55, format='%01.3f')
            pp.savefig(rd + '/U_%s.png' % j, bbox_inches='tight', dpi=600)
            pp.close()
            ax = plot(Pkp)
            pp.colorbar(ax, shrink=0.55, format='%01.3f')
            pp.savefig(rd + '/P_%s.png' % j, bbox_inches='tight', dpi=600)
            pp.close()
        else:
            _export_ply(mesh, int_cells, Uk, j, False, 1 / 255, rd, flog)

        total_time += time.time() - start_iteration_time
        export = [j, total_time, energy[j], opt[j], rel_change[j]]
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
    tolerance = eval(config.get('GCG', 'tolerance', fallback='1e-12'))

    _main_()
