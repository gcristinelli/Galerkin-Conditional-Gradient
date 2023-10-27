import numpy as np
from numpy import linalg as la


def _SSN(Kl, Km, coefficients, mean, measurements, alpha, M):
    if len(coefficients) == 0:
        tol = 1e-14
        H = Km.T @ M @ Km
        Id = np.identity(coefficients.size + 1)
        length = coefficients.size
        K = Km
        theta = 1e-9
        # Setup initial q
        point = mean
        misfit = K @ point - measurements
        q = point - Km.T @ M @ misfit
        point = np.append(np.maximum(q[:length], 0), q[length])

        # Iterate
        for i in range(1000):

            misfit = K @ point - measurements
            righthand = q - point + Km.T @ M @ misfit
            if la.norm(righthand) <= tol:
                coefficients = np.maximum(q[:length], 0)
                mean = np.array([q[length]])
                adjoint = -Km.T @ M @ misfit
                F_val = 0.5 * np.dot(misfit, M @ misfit)
                optval = 0.5 * np.dot(misfit, M @ misfit) + alpha * la.norm(point[:length], 1)
                print("SSN terminated with residual", la.norm(righthand), "after", i, "iterations")
                break

            theta = theta / 10
            direc = la.solve(H + theta * Id, righthand)
            qnew = q - direc
            newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
            newmisfit = K @ newpoint - measurements
            qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) - 0.5 * np.dot(misfit, M @ misfit)
            while qdiff >= 1e-3:
                theta = 2 * theta
                direc = la.solve(H + theta * Id, righthand)
                qnew = q - direc
                newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
                newmisfit = K @ newpoint - measurements
                qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) - 0.5 * np.dot(misfit, M @ misfit)

            q = qnew
            point = newpoint


    else:
        tol = 1e-14
        H = np.block([[Kl.T @ M @ Kl, Kl.T @ M @ Km], [Km.T @ M @ Kl, Km.T @ M @ Km]])
        Id = np.identity(coefficients.size + 1)
        length = coefficients.size
        K = np.concatenate((Kl, Km), axis=1)
        theta = 1e-9
        # Setup initial q
        point = np.concatenate((coefficients, mean))
        misfit = K @ point - measurements
        q = point - np.concatenate((Kl.T @ M @ misfit + alpha, Km.T @ M @ misfit))
        point = np.append(np.maximum(q[:length], 0), q[length])

        # Iterate
        for i in range(1000):

            misfit = K @ point - measurements
            righthand = q - point + np.concatenate((Kl.T @ M @ misfit + alpha, Km.T @ M @ misfit))
            if la.norm(righthand) <= tol:
                coefficients = np.maximum(q[:length], 0)
                mean = np.array([q[length]])
                adjoint = -np.concatenate((Kl.T @ M @ misfit, Km.T @ M @ misfit))
                F_val = 0.5 * np.dot(misfit, M @ misfit)
                optval = 0.5 * np.dot(misfit, M @ misfit) + alpha * la.norm(point[:length], 1)
                print("SSN terminated with residual", la.norm(righthand), "after", i, "iterations")
                break
            D = np.diag(np.append(np.where(q[:length] > 0, 1, 0), 1))
            Mo = Id - D + H @ D
            theta = theta / 10
            direc = la.solve(Mo + theta * Id, righthand)
            qnew = q - direc
            newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
            newmisfit = K @ newpoint - measurements
            qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) + alpha * la.norm(newpoint[:length], 1) \
                    - 0.5 * np.dot(misfit, M @ misfit) - alpha * la.norm(point[:length], 1)
            while qdiff >= 1e-3:
                theta = 2 * theta
                direc = la.solve(Mo + theta * Id, righthand)
                qnew = q - direc
                newpoint = np.append(np.maximum(qnew[:length], 0), qnew[length])
                newmisfit = K @ newpoint - measurements
                qdiff = 0.5 * np.dot(newmisfit, M @ newmisfit) + alpha * la.norm(newpoint[:length], 1) \
                        - 0.5 * np.dot(misfit, M @ misfit) - alpha * la.norm(point[:length], 1)

            q = qnew
            point = newpoint

    return coefficients, mean, adjoint, optval, F_val
