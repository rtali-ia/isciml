
##################################################################################
# This code is used transfer matrices from Python Assembly to Ginkgo Solver.
# Ginkgo requires the A matrix, b vector and a guess vector x0.
# A, b, x0 have to be converted to ".mtx" files. Ginkgo understands these files.
##################################################################################

##
# Assume A matrix is stored as val_A
##

A_coo = sp.coo_matrix((val_A, (row_id_A, col_id_A)),
                          shape=(n_in_edge, n_in_edge))
A_csc = A_coo.tocsc()

##
# Assume the dense B vector is stored as val_b
##

#Output .mtx files
# Write the matrix to a mmt file
target = 'FE_stiffness_matrix.mtx'
mmwrite(target, A_coo)

vect = 'FE_stiffness_vector.mtx'
mmwrite(vect, val_b)

# Create the initial guess with unit norm and zero phase
guess = (1/np.sqrt(n_in_edge))*np.ones((n_in_edge,), dtype=complex)
start = 'x0.mtx'
mmwrite(start, guess)
