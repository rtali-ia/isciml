# Mesh file with M tetra and N nodes info (material property defined elsewhere)

vtk_filepath='../Mesh_Files/Hormozblocks3d.vtk' 
# M x 2 .npy file. 1st column contains the M sigma values (S/m), 2nd column contains M mu_r values (non_magnetic mu_r=1.0)

# cond_mu_filepath='/home/exouser/FEFD_Prototype/Input_Models/sig_mu_Hormoz.npy' 
cond_mu_filepath='../Input_Models/Halfspace_02_Hormozblocks3d.npy' 

# File containing node indices for transmitter wire segments. 
# Must have run the data prep code Parallelized_Create_and_Prep_Data_for_FEFD.py or Create_and_Prep_Data_for_FEFD.py scripts first.
txseg_filepath='../Tx_Rx_Files/TxSegFiles/wire_vector_blocks3d_txsegs.csv'

# Must be 3 columns csv format, no headings. The x,y,z coordinates of receiver locations (in m).
# The generated receiver points are processed to make sure they are inside a tetrahedron and not on a face, node, or edge.
# 1 mm is usually added to a one of the coordinates to push them inside the tetrahedron. 

# Receiver locations. 3 column csv file with header denoting x-, y-, z- columns. (right handed coordinate system withz positive upwards convention)
rxcoord_folder='../Tx_Rx_Files/'
rxcoord_filename='rxpts1mmdown_blocks3d' # Don't put .csv or extension name.
xyzcolnames=['x (m)','y (m)','z (m)']

# Folder containing Tetra indices for receiver locations. 
# Must have run the data prep code Parallelized_Create_and_Prep_Data_for_FEFD.py or Create_and_Prep_Data_for_FEFD.py scripts first.
rxtet_folder='../Tx_Rx_Files/RxTetraFiles/'

# Single column csv file no heading. Contains the elements of tetrahedra containing the receviver locations.

# Directory and filename of exported data. 6 column csv file. WILL CONTAIN HEADERS.
# The columns are organized as x-, y-, z-, of receiver locations (in m), followed by real Ex, imag Ex, real Ey, imag Ey, real Ez, imag Ez, all in V/m

export_filepath='..Output_Data/Efield_Halfspace_0p02_f30000_blocks3d.csv'

# Frequency in Hz
f=30000.

# Peak current strength
Io=1. 

# tolerance limit to determine coordinate value coincidence in space
tol_thresh_space=1.e-4