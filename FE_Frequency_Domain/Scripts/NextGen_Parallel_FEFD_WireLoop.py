import numpy as np
import pandas as pd
from numpy.linalg import inv
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
# from scikits.umfpack import spsolve
# from joblib import Parallel, delayed
from multiprocessing import cpu_count
from multiprocessing import Pool
import time

def load_mesh_files(in_vtk):
    # in_vtk=file_dir+base_file+'.vtk'
    txtopen = open(in_vtk,'r')
    all_vtk_lines = txtopen.readlines()
    txtopen.close()
    skp_hdr_ln_no = 3 # Line no. 4
    n_pts_ln_no = skp_hdr_ln_no + 1 # This line contains information about the number of nodes
    npts = int(all_vtk_lines[n_pts_ln_no].split()[1]) # The number of nodes in the mesh

    ndes = np.zeros((npts,3), dtype=float)

    for i_pt in range(0, npts):
        ndes[i_pt,:] = list(map(float, all_vtk_lines[n_pts_ln_no+1+i_pt].split()[0:])) # Read in the coordinates of the nodes
    
    cells_ln_no = n_pts_ln_no + 1 + npts + 1 # This line contains information about the number of tetrahedral cells.
    ncells = int(all_vtk_lines[cells_ln_no].split()[1]) # Get the value of the number of cells (tetrahedra) in the mesh

    tet_ndes = np.zeros((ncells, 4), dtype='int')

    for i_cell in range(0, ncells):
        tet_ndes[i_cell,:] = list(map(int, all_vtk_lines[cells_ln_no+1+i_cell].split()[1:])) # Read in the node indices (starting from 0) for individual tetrahedron
        
    return npts, ncells, ndes, tet_ndes

def edge_belongs_to(bnd_i, edgetotal, bedge_tf):
    # Given an array of J X 1 integer valued boundary node indices in b_nd_i, b_edge_tf returns 0 or 1 
    # depending on whether an edge from the Q X 2 array of edge_total belongs to that boundary. 

    for iedge, edge_i in enumerate(edgetotal):

        edg_tail = edge_i[0]
        edg_tip = edge_i[1]

        chk_tail = np.any(bnd_i == edg_tail)
        chk_tip = np.any(bnd_i == edg_tip)

        if chk_tail & chk_tip:

            bedge_tf[iedge] = 1
        else:
            bedge_tf[iedge] = 0

    return bedge_tf

def find_node_index(ndes, wvec):
    # Determine the index number of a node coordinate located in the W X 3 wv array by comparing it to the 
    # N X 3 global node coordinate array, nodes
    nd_w = wvec.shape[0]
    wi = np.zeros(nd_w, dtype = int)

    for iw, w_cord in enumerate(wvec):

        indw = np.where((np.abs(ndes[:,0] - w_cord[0]) <= tol_thresh_space) & 
                        (np.abs(ndes[:,1] - w_cord[1]) <= tol_thresh_space) &
                        (np.abs(ndes[:,2] - w_cord[2]) <= tol_thresh_space))[0]
        wi[iw] = indw
    return wi

def bary_coord(rxpts, n_a, n_k, n_l, n_m, n_n, ndes):
    # Determine the position of the bary_centric coordinate of a tetrahedral node, na, at the receiver point, rx_pt in a
    # tetrahedron defined by node indices, nk, nl, nm, nn, where nk < nl < nm < nn.
    x = rxpts[0]
    y = rxpts[1]
    z = rxpts[2]

    x1 = ndes[n_k, 0]
    y1 = ndes[n_k, 1]
    z1 = ndes[n_k, 2]
    x2 = ndes[n_l, 0]
    y2 = ndes[n_l, 1]
    z2 = ndes[n_l, 2]
    x3 = ndes[n_m, 0]
    y3 = ndes[n_m, 1]
    z3 = ndes[n_m, 2]
    x4 = ndes[n_n, 0]
    y4 = ndes[n_n, 1]
    z4 = ndes[n_n, 2]

    d0 = np.linalg.det([[x1,y1,z1,1.], [x2,y2,z2,1.], [x3,y3,z3,1.], [x4,y4,z4,1.]])
    if   n_a == n_k:
        di = np.linalg.det([[x,y,z,1.],[x2,y2,z2,1.],[x3,y3,z3,1.],[x4,y4,z4,1.]])
        return di/d0
    elif n_a == n_l:
        di = np.linalg.det([[x1,y1,z1,1.],[x,y,z,1.],[x3,y3,z3,1.],[x4,y4,z4,1.]])
        return di/d0
    elif n_a == n_m:
        di = np.linalg.det([[x1,y1,z1,1.],[x2,y2,z2,1.],[x,y,z,1.],[x4,y4,z4,1.]])
        return di/d0
    elif n_a == n_n:
        di = np.linalg.det([[x1,y1,z1,1.],[x2,y2,z2,1.],[x3,y3,z3,1.],[x,y,z,1.]])
        return di/d0
    else:
        raise ValueError('node ', n_a, ' is not represented by', n_k, n_l, n_m, n_n, '. Please check.')
    
def grad_na(n_a, n_k, n_l, n_m, n_n, g_nk, g_nl, g_nm, g_nn):
    # calculate the gradient of a node n_a, for a tetrahedron with nodes, n_k, n_l, n_m, n_n
    # and pre computed gradients g_nk, g_nl, g_nm, g_nn.

    if n_a == n_k:
        return g_nk
    if n_a == n_l:
        return g_nl
    if n_a == n_m:
        return g_nm
    if n_a == n_n:
        return g_nn

def crl_edge(g_na, g_nb):
    # calculate the curl of an edge vector, directed from node n_a, to node n_b. 
    # The gradients of these nodes are g_na, and g_nb respectively.
    c_g = np.cross(g_na, g_nb)
    return 2.0*c_g

def crl_dot_crl(g_na, g_nb, g_nc, g_nd, det_rk):
    # calculate the dot product of the curl of two edge vectors [n_a, n_b] and [n_c, n_d]. 
    # Their gradients are g_na, g_nb, g_nc, and g_nd, respectively.
    # det_rk is the absolute value of the transformation determinant as defined in Velimsky (2003)
    crl_i = crl_edge(g_na, g_nb) ; crl_j = crl_edge(g_nc, g_nd)
    return det_rk*(np.dot(crl_i, crl_j))/6.0

def delfn(n_1, n_2):
    # The delta function that returns 1 if n_1 == n_2, and 0 otherwise.
    if n_1 == n_2:
        return 1
    else:
        return 0

def edg_dot_edg(e_i, g_na, g_nb, e_j, g_nc, g_nd, det_rk):
    # Calculate the dot products of the edge vectors, ei, and ej, whose nodes are [na, nb] and [nc, nd].
    # The corresponding node gradients are gna, gnb, gnc, and gnd, respectively.

    n_a = e_i[0]
    n_b = e_i[1]
    n_c = e_j[0]
    n_d = e_j[1]

    d_ac = delfn(n_a,n_c)
    d_ad = delfn(n_a,n_d)
    d_bc = delfn(n_b,n_c)
    d_bd = delfn(n_b,n_d)

    t1 = (1+d_ac)*(np.dot(g_nb,g_nd))
    t2 = (1+d_ad)*(np.dot(g_nb,g_nc))
    t3 = (1+d_bc)*(np.dot(g_na,g_nd))
    t4 = (1+d_bd)*(np.dot(g_na,g_nc))

    return (det_rk/120.0)*(t1 - t2 - t3 + t4)

def inv_mat3x3(r_k, det_rk):
    # Calculate the inverse of a 3 X 3 matrix. 
    # This is to override error messages when python evaluates the determinant to be zero.
    rk_inv = np.zeros((3, 3))
    if det_rk == 0.0:
        raise ValueError('Determinant ', det_rk, '. Matrix is singular, please check.')
    else:
        co11=r_k[1,1]*r_k[2,2]-r_k[1,2]*r_k[2,1]
        co12=r_k[1,0]*r_k[2,2]-r_k[1,2]*r_k[2,0]
        co13=r_k[1,0]*r_k[2,1]-r_k[1,1]*r_k[2,0]
        co21=r_k[0,1]*r_k[2,2]-r_k[0,2]*r_k[2,1]
        co22=r_k[0,0]*r_k[2,2]-r_k[0,2]*r_k[2,0]
        co23=r_k[0,0]*r_k[2,1]-r_k[0,1]*r_k[2,0]
        co31=r_k[0,1]*r_k[1,2]-r_k[0,2]*r_k[1,1]
        co32=r_k[0,0]*r_k[1,2]-r_k[0,2]*r_k[1,0]
        co33=r_k[0,0]*r_k[1,1]-r_k[0,1]*r_k[1,0]

        rk_inv[0,0] =  co11/det_rk 
        rk_inv[0,1] = -co21/det_rk
        rk_inv[0,2] =  co31/det_rk
        rk_inv[1,0] = -co12/det_rk 
        rk_inv[1,1] =  co22/det_rk
        rk_inv[1,2] = -co32/det_rk
        rk_inv[2,0] =  co13/det_rk
        rk_inv[2,1] = -co23/det_rk
        rk_inv[2,2] =  co33/det_rk

    return rk_inv

def compute_crlcrl_edgedg(e_i, e_j, n_k, n_l, n_m, n_n, g_nk, g_nl, g_nm, g_nn, det_rk):
    # Find the indices ei_ind, and ej_ind of edge vectors, ei and ej in the global inner edge vector array in_edge. 
    # In addition, compute the edge - edge dot product and curl dot curl products of the edge vectors for a given tetrahedron.
    # The nodal gradients of the tetrahedron are gnk, gnl, gnm, and gnn, for nodes nk, nl, nm, and nn.
    # detrk is the absolute value of the transformation determinant defined in Velimsky (2003).

    #ei_ind = np.where((in_edge[:,0] == ei[0]) & (in_edge[:,1] == ei[1]))[0][0]
    #ej_ind = np.where((in_edge[:,0] == ej[0]) & (in_edge[:,1] == ej[1]))[0][0]

    n_a = e_i[0]
    n_b = e_i[1]
    n_c = e_j[0]
    n_d = e_j[1]

    g_na = grad_na(n_a,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    g_nb = grad_na(n_b,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    g_nc = grad_na(n_c,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    g_nd = grad_na(n_d,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)

    ci_cj = crl_dot_crl(g_na,g_nb,g_nc,g_nd,det_rk)
    ei_ej = edg_dot_edg(e_i,g_na,g_nb,e_j,g_nc,g_nd,det_rk)

    return ci_cj, ei_ej


def compute_diag_tet(e_i, n_k, n_l, n_m, n_n, g_nk, g_nl, g_nm, g_nn, det_rk):
    # Find the index ei_ind of an edge vector, ei in the global inner edge vector array in_edge. 
    # In addition, compute the edge - edge dot product and curl edge dot curl edge products of the edge vector for a given tetrahedron.
    # The nodal gradients of the tetrahedron are gnk, gnl, gnm, and gnn, for nodes nk, nl, nm, and nn.
    # detrk is the absolute value of the transformation determinant defined in Velimsky (2003).
    #ei_ind = np.where((in_edge[:,0] == ei[0]) & (in_edge[:,1] == ei[1]))[0][0]
    n_a = e_i[0] ; n_b = e_i[1]
    g_na = grad_na(n_a,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    g_nb = grad_na(n_b,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    ci_ci = crl_dot_crl(g_na,g_nb,g_na,g_nb,det_rk)
    ei_ei = edg_dot_edg(e_i,g_na,g_nb,e_i,g_na,g_nb,det_rk)
    return ci_ci, ei_ei

def compute_offdiag_tet(e_i, e_j, n_k, n_l, n_m, n_n, g_nk, g_nl, g_nm, g_nn, det_rk):
    # Find the indices ei_ind, and ej_ind of edge vectors, ei and ej in the global inner edge vector array in_edge. 
    # In addition, compute the edge - edge dot product and curl dot curl products of the edge vectors for a given tetrahedron.
    # The nodal gradients of the tetrahedron are gnk, gnl, gnm, and gnn, for nodes nk, nl, nm, and nn.
    # detrk is the absolute value of the transformation determinant defined in Velimsky (2003).

    #ei_ind = np.where((in_edge[:,0] == ei[0]) & (in_edge[:,1] == ei[1]))[0][0]
    #ej_ind = np.where((in_edge[:,0] == ej[0]) & (in_edge[:,1] == ej[1]))[0][0]

    n_a = e_i[0]
    n_b = e_i[1]
    n_c = e_j[0]
    n_d = e_j[1]

    g_na = grad_na(n_a,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    g_nb = grad_na(n_b,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    g_nc = grad_na(n_c,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
    g_nd = grad_na(n_d,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)

    ci_cj = crl_dot_crl(g_na,g_nb,g_nc,g_nd,det_rk)
    ei_ej = edg_dot_edg(e_i,g_na,g_nb,e_j,g_nc,g_nd,det_rk)

    return ci_cj, ei_ej

def int_edge_wire_seg(e_k, w_i, n_k, n_l, n_m, n_n, g_nk, g_nl, g_nm, g_nn):

    w_c=wiresegs[w_i,0]
    w_d=wiresegs[w_i,1]
    n_a=e_k[0]
    n_b=e_k[1]

    delac=delfn(n_a, w_c)
    delbc=delfn(n_b, w_c)
    delad=delfn(n_a, w_d)
    delbd=delfn(n_b, w_d)

    wnc=delac+delbc+delad+delbd

    if wnc > 0:
        gr_na = grad_na(n_a,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
        gr_nb = grad_na(n_b,n_k,n_l,n_m,n_n,g_nk,g_nl,g_nm,g_nn)
        js = nodes[w_d,:]-nodes[w_c,:]
        intnw1 = delac*(np.dot(gr_nb,js))-delbc*(np.dot(gr_na,js))
        intnw2 = delad*(np.dot(gr_nb,js))-delbd*(np.dot(gr_na,js))
        intnw = intnw1 + intnw2
    else:
        intnw=0
    
    return intnw

def select_tet_edg_vector(edgind,e_1,e_2,e_3,e_4,e_5,e_6):
    if edgind==0:
        edgval=e_1
    elif edgind==1:
        edgval=e_2
    elif edgind==2:
        edgval=e_3
    elif edgind==3:
        edgval=e_4
    elif edgind==4:
        edgval=e_5
    else:
        edgval=e_6
    
    return edgval

def create_sparse_initial_FE_matrix_vector(itet):
    tet_i=tetnodes[itet]
    
    nk = tet_i[0]
    nl = tet_i[1]
    nm = tet_i[2]
    nn = tet_i[3]
    
    r1 = nodes[nl,:] - nodes[nk,:]
    r2 = nodes[nm,:] - nodes[nk,:]
    r3 = nodes[nn,:] - nodes[nk,:]
    
    rkp = np.array([r1, r2, r3])
    rk = rkp.T

    detrk = np.abs(np.linalg.det(rk))
    
    if detrk == 0.0:
        detrk = 1e-8
    else:
        detrk = 1.0*detrk
    
    rki = inv_mat3x3(rk, detrk)
    rkit = rki.T
    # tet_rkdet.append(detrk)

    # Using the inverse transpose of the reference matrix to calculate the actual gradients of tetrahedral nodes. 
    # Refer Velimsky (2003) for more details
    
    gnk = np.matmul(rkit,gk)
    gnl = np.matmul(rkit,gl)
    gnm = np.matmul(rkit,gm)
    gnn = np.matmul(rkit,gn)

    # tet_gnk.append(list(gnk))
    # tet_gnl.append(list(gnl))
    # tet_gnm.append(list(gnm))
    # tet_gnn.append(list(gnn))
    
    e1 = edge1[itet]
    e2 = edge2[itet]
    e3 = edge3[itet]
    e4 = edge4[itet]
    e5 = edge5[itet]
    e6 = edge6[itet]

    w1 = wire_edge_tet[itet,0]
    w2 = wire_edge_tet[itet,1]
    w3 = wire_edge_tet[itet,2]
    w4 = wire_edge_tet[itet,3]
    w5 = wire_edge_tet[itet,4]
    w6 = wire_edge_tet[itet,5]

    elidPQ=[]
    elidRHS=[]
    for iedg in range(6):
        if (in_edge_tet[itet,iedg] > -1):
            ei=select_tet_edg_vector(iedg,e1,e2,e3,e4,e5,e6)
            ei_ind = in_edge_tet[itet, iedg]
            cici, eiei = compute_crlcrl_edgedg(ei, ei, nk, nl, nm, nn, gnk, gnl, gnm, gnn, detrk)
            P_fillterm = sig_tet[itet]*eiei*le[ei_ind]*le[ei_ind]*mu0
            Q_fillterm = nu_tet[itet]*cici*le[ei_ind]*le[ei_ind]
            elid = ei_ind*n_in_edge + ei_ind

            elidPQ.append(('LHS',elid, P_fillterm, Q_fillterm))
                
            for jedg in range(iedg+1,6):
                if (in_edge_tet[itet,jedg] > -1):
                    ej=select_tet_edg_vector(jedg,e1,e2,e3,e4,e5,e6)
                    ej_ind = in_edge_tet[itet, jedg]
                    cicj, eiej = compute_crlcrl_edgedg(ei, ej, nk, nl, nm, nn, gnk, gnl, gnm, gnn, detrk)
                    P_fillterm = sig_tet[itet]*eiej*le[ei_ind]*le[ej_ind]*mu0
                    Q_fillterm = nu_tet[itet]*cicj*le[ei_ind]*le[ej_ind]
                    elidij = ei_ind*n_in_edge + ej_ind
                    elidji = ej_ind*n_in_edge + ei_ind

                    elidPQ.append(('LHS',elidij, P_fillterm, Q_fillterm))
                    elidPQ.append(('LHS',elidji, P_fillterm, Q_fillterm))

        # Filling up RHS
            
            if w1 > -1:
                inteiwk=int_edge_wire_seg(ei, w1, nk, nl, nm, nn, gnk, gnl, gnm, gnn)
                rhs_fillterm=Io*inteiwk*le[ei_ind]*mu0
                elid_rhs = ei_ind
                elidRHS.append(('RHS',elid_rhs,rhs_fillterm))
                
            if w2 > -1:
                inteiwk=int_edge_wire_seg(ei, w2, nk, nl, nm, nn, gnk, gnl, gnm, gnn)
                rhs_fillterm=Io*inteiwk*le[ei_ind]*mu0
                elid_rhs = ei_ind
                elidRHS.append(('RHS',elid_rhs,rhs_fillterm))

            if w3 > -1:
                inteiwk=int_edge_wire_seg(ei, w3, nk, nl, nm, nn, gnk, gnl, gnm, gnn)
                rhs_fillterm=Io*inteiwk*le[ei_ind]*mu0
                elid_rhs = ei_ind
                elidRHS.append(('RHS',elid_rhs,rhs_fillterm))

            if w4 > -1:
                inteiwk=int_edge_wire_seg(ei, w4, nk, nl, nm, nn, gnk, gnl, gnm, gnn)
                rhs_fillterm=Io*inteiwk*le[ei_ind]*mu0
                elid_rhs = ei_ind
                elidRHS.append(('RHS',elid_rhs,rhs_fillterm))

            if w5 > -1:
                inteiwk=int_edge_wire_seg(ei, w5, nk, nl, nm, nn, gnk, gnl, gnm, gnn)
                rhs_fillterm=Io*inteiwk*le[ei_ind]*mu0
                elid_rhs = ei_ind
                elidRHS.append(('RHS',elid_rhs,rhs_fillterm))

            if w6 > -1:
                inteiwk=int_edge_wire_seg(ei, w6, nk, nl, nm, nn, gnk, gnl, gnm, gnn)
                rhs_fillterm=Io*inteiwk*le[ei_ind]*mu0
                elid_rhs = ei_ind
                elidRHS.append(('RHS',elid_rhs,rhs_fillterm))

    return ('Tet_grad_det',list(gnk),list(gnl),list(gnm),list(gnn),detrk), elidPQ, elidRHS

def Efield_postproc(irec):
    rx_pt=reccoords[irec]

    E_rx=np.array([0., 0., 0.])+1j*np.array([0., 0., 0.])

    dfrxtet=pd.read_csv(rxtet_folder+rxcoord_filename+'_rxtetra_'+str(irec)+'.csv')
    rxtetra=dfrxtet.to_numpy()

    nrxtet=len(rxtetra)

    for itet in range(nrxtet):
        tet_i=rxtetra[itet,0]
        e1 = edge1[tet_i]
        e2 = edge2[tet_i]
        e3 = edge3[tet_i]
        e4 = edge4[tet_i]
        e5 = edge5[tet_i]
        e6 = edge6[tet_i]

        nk = tetnodes[tet_i,0]
        nl = tetnodes[tet_i,1]
        nm = tetnodes[tet_i,2]
        nn = tetnodes[tet_i,3]

        gnk = tet_gnk[tet_i]
        gnl = tet_gnl[tet_i]
        gnm = tet_gnm[tet_i]
        gnn = tet_gnn[tet_i]

        for medg in range(6):
            if (in_edge_tet[tet_i,medg] > -1):
                e_m=select_tet_edg_vector(medg,e1,e2,e3,e4,e5,e6)
            # m_indx=np.where((e_m[0]==in_edge[:,0]) & (e_m[1]==in_edge[:,1]))[0]
            # if len(m_indx) > 0:
                na = e_m[0]
                nb = e_m[1]
                edg_m = in_edge_tet[tet_i,medg]
                gna = np.array(grad_na(na,nk,nl,nm,nn,gnk,gnl,gnm,gnn))
                gnb = np.array(grad_na(nb,nk,nl,nm,nn,gnk,gnl,gnm,gnn))
                b_na = bary_coord(rx_pt,na,nk,nl,nm,nn,nodes)
                b_nb = bary_coord(rx_pt,nb,nk,nl,nm,nn,nodes)
                E_rx = E_rx + FE_x[edg_m]*(b_na*gnb - b_nb*gna)*le[edg_m]
                # E_rx[irx] = E_rx[irx] + FE_x[edg_m]*(b_na*gnb - b_nb*gna)*le[edg_m]
                # E_rx = E_rx + FE_x[m_indx]*(b_na*gnb - b_nb*gna)*le[m_indx]
                
    return [rx_pt[0],rx_pt[1],rx_pt[2],np.real(E_rx[0]/nrxtet),np.imag(E_rx[0]/nrxtet),np.real(E_rx[1]/nrxtet),np.imag(E_rx[1]/nrxtet),np.real(E_rx[2]/nrxtet),np.imag(E_rx[2]/nrxtet)]

def postprocBrec(irec):
    rx_pt=reccoords[irec]
    # B_rx=np.array([0., 0., 0.])+1j*np.array([0., 0., 0.])

    miwBz=0.0+1j*0.0

    dfrxtet=pd.read_csv(rxtet_folder+rxcoord_filename+'_rxtetra_'+str(irec)+'.csv')
    rxtetra=dfrxtet.to_numpy()

    nrxtet=len(rxtetra)

    for itet in range(nrxtet):
        tet_i=rxtetra[itet,0]
        e1 = edge1[tet_i]
        e2 = edge2[tet_i]
        e3 = edge3[tet_i]
        e4 = edge4[tet_i]
        e5 = edge5[tet_i]
        e6 = edge6[tet_i]

        nk = tetnodes[tet_i,0]
        nl = tetnodes[tet_i,1]
        nm = tetnodes[tet_i,2]
        nn = tetnodes[tet_i,3]

        gnk = tet_gnk[tet_i]
        gnl = tet_gnl[tet_i]
        gnm = tet_gnm[tet_i]
        gnn = tet_gnn[tet_i]

        for medg in range(6):
            if (in_edge_tet[tet_i,medg] > -1):
                e_m=select_tet_edg_vector(medg,e1,e2,e3,e4,e5,e6)
            # m_indx=np.where((e_m[0]==in_edge[:,0]) & (e_m[1]==in_edge[:,1]))[0]
            # if len(m_indx) > 0:
                na = e_m[0]
                nb = e_m[1]
                edg_m = in_edge_tet[tet_i,medg]
                gna = np.array(grad_na(na,nk,nl,nm,nn,gnk,gnl,gnm,gnn))
                gnb = np.array(grad_na(nb,nk,nl,nm,nn,gnk,gnl,gnm,gnn))
                crl_edg_m = crl_edge(gna, gnb)
                acrle = FE_x[edg_m]*crl_edg_m*le[edg_m]
                miwBz=miwBz-acrle[2]
                
    miwBzf=miwBz/nrxtet
    Bz = 1j*(miwBzf/omg)

    return [rx_pt[0],rx_pt[1],rx_pt[2],np.real(Bz),np.imag(Bz),np.real(miwBzf),np.imag(miwBzf)]

def create_local_edge_tetmap(loc_indx):

    loc_eg_i_inv = (-1)*np.ones(len(edge_total), dtype=int)

    for ix, loc_eg_i_vals in enumerate(loc_indx):
        loc_eg_i_inv[loc_eg_i_vals] = ix

    loc_edge_tet = np.zeros((ntet, 6), dtype='int64')

    edg_tot_inv_i_loc= np.zeros(len(edg_tot_inv_i), dtype=int)

    for ix, edg_glb_i in enumerate(edg_tot_inv_i):
        edg_tot_inv_i_loc[ix] = loc_eg_i_inv[edg_glb_i]

    loc_edge_tet[:,0] = edg_tot_inv_i_loc[0:ntet]
    loc_edge_tet[:,1] = edg_tot_inv_i_loc[ntet:2*ntet]
    loc_edge_tet[:,2] = edg_tot_inv_i_loc[2*ntet:3*ntet]
    loc_edge_tet[:,3] = edg_tot_inv_i_loc[3*ntet:4*ntet]
    loc_edge_tet[:,4] = edg_tot_inv_i_loc[4*ntet:5*ntet]
    loc_edge_tet[:,5] = edg_tot_inv_i_loc[5*ntet:6*ntet]

    return loc_edge_tet

if __name__ == '__main__':
    from inpt_NextGen_Parallel_FEFD_WireLoop import *
    start_time=time.time()

    pi=np.pi
    mu0=4.*pi*1e-7
    omg=2.*pi*f

    nnodes, ntet, nodes, tet_no_sort=load_mesh_files(vtk_filepath)
    cond_mu=np.load(cond_mu_filepath)

    sig_tet=cond_mu[:,0]
    nu_tet=1./cond_mu[:,1]
    
    # Sorting nodes in a tetrahedra in ascending order (axis=1: sort in the horizontal direction)
    tetnodes = np.sort(tet_no_sort, axis=1)

    xmin=np.min(nodes[:,0])
    xmax=np.max(nodes[:,0])
    ymin=np.min(nodes[:,1])
    ymax=np.max(nodes[:,1])
    zmin=np.min(nodes[:,2])
    zmax=np.max(nodes[:,2])

    # determine nodes on the boundary

    bndi1=np.where((np.abs(nodes[:,0]-xmin) < 1.e-4) | (np.abs(nodes[:,0]-xmax) < 1.e-4))[0]
    bndi2=np.where((np.abs(nodes[:,1]-ymin) < 1.e-4) | (np.abs(nodes[:,1]-ymax) < 1.e-4))[0]
    bndi3=np.where((np.abs(nodes[:,2]-zmin) < 1.e-4) | (np.abs(nodes[:,2]-zmax) < 1.e-4))[0]

    b_nd_i_gross=np.hstack((bndi1,bndi2,bndi3))
    b_nd_i=np.unique(b_nd_i_gross)

    # (nd1: the first column of 'tetnodes', as a column vector)
    nd1 = tetnodes[:,0].reshape(ntet,1)
    nd2 = tetnodes[:,1].reshape(ntet,1) 
    nd3 = tetnodes[:,2].reshape(ntet,1)
    nd4 = tetnodes[:,3].reshape(ntet,1)

    # Defining the 6 edges of a tetrahedron for all tetrahedra (hstack: horizontal stacking of column vectors)
    edge1 = np.hstack((nd1, nd2))
    edge2 = np.hstack((nd1, nd3))
    edge3 = np.hstack((nd1, nd4))
    edge4 = np.hstack((nd2, nd3))
    edge5 = np.hstack((nd2, nd4))
    edge6 = np.hstack((nd3, nd4))

    # Form the global edges vector (concatenate in axis=0 direction: similar to vertical stacking)
    # (unique: remove multiple occurance of node-pairs ==> results in a global array)edge_total = np.unique(np.concatenate((edge1,edge2,edge3,edge4,edge5,edge6),axis=0), axis = 0)
    edge_total, edg_tot_inv_i = np.unique(np.concatenate((edge1, edge2, edge3, edge4, edge5, edge6), axis=0 ),axis=0, return_inverse=True)
    
    # the no of rows of 'edge_total'
    nedges = edge_total.shape[0]

    # an empty array of size 'nedges'
    b_edge_tf = np.zeros(nedges, dtype=int)

    # identify edges belonging to boundary ('b_edge_tf' contains 1 or 0, for being on the bd or not)
    b_edge_tf = edge_belongs_to(b_nd_i, edge_total, b_edge_tf)

    # "inner" edges not on boundary. This is the array containing the indices of the inner edges 

    in_eg_i = np.where(b_edge_tf == 0)[0]

    # the nodes of the internal edges
    in_edge = edge_total[in_eg_i, :]

    # total number of internal edges is the no of complex unknowns
    n_in_edge = in_edge.shape[0]

    # the array for the length of the internal edges
    le = np.zeros(n_in_edge)

    xe = nodes[in_edge[:, 1], 0] - nodes[in_edge[:, 0], 0]
    ye = nodes[in_edge[:, 1], 1] - nodes[in_edge[:, 0], 1]
    ze = nodes[in_edge[:, 1], 2] - nodes[in_edge[:, 0], 2]

    le = np.sqrt(xe**2 + ye**2 + ze**2)

    in_edge_tet=create_local_edge_tetmap(in_eg_i)

    # Set up the source. Load the wire segment indices
    dftxsegs=pd.read_csv(txseg_filepath)
    wiresegs=dftxsegs.to_numpy()

    wire_edg_i=np.where((edge_total == wiresegs[:, None]).all(-1))[1]
    wire_edge_tet=create_local_edge_tetmap(wire_edg_i)
    # nwseg=len(wiresegs)

    # Defining the gradients of the nodes of the reference tetrahedron (Velimsky, 2003)

    gk=np.array([-1, -1, -1])
    gl=np.array([1, 0, 0])
    gm=np.array([0,  1,  0])
    gn=np.array([0, 0, 1])

    n_cpu = cpu_count()

    print(n_cpu,' CPUs available')

    job_cpus=int(n_cpu/2)

    print(job_cpus,' CPUs used')
    tet_indices=list(range(ntet))

    # Filling up LHS and RHS of FE stiffness matrix

    tic1 = time.time()

    # tet_info = Parallel(n_jobs=job_cpus)(delayed(create_sparse_initial_FE_matrix_vector)(itet) for itet in tet_indices)
    with Pool(processes=job_cpus) as pool:
        tet_info=list(pool.map(create_sparse_initial_FE_matrix_vector, tet_indices))

    tic2=time.time()

    print('Completed tet organization in ',tic2-tic1,' seconds. Used ',job_cpus,' of ',n_cpu,' CPUs.')

    elid_val_P={}
    elid_val_Q={}
    elid_val_s1={}

    tet_rkdet=[]
    tet_gnk=[]
    tet_gnl=[]
    tet_gnm=[]
    tet_gnn=[]

    for itet in range(ntet):
        tetgraddata=tet_info[itet][0]
        tet_gnk.append(tetgraddata[1])
        tet_gnl.append(tetgraddata[2])
        tet_gnm.append(tetgraddata[3])
        tet_gnn.append(tetgraddata[4])
        tet_rkdet.append(tetgraddata[5])

        tetLHSdata=tet_info[itet][1]
        if len(tetLHSdata) > 0:
            for iLHS in range(len(tetLHSdata)):
                elid=tetLHSdata[iLHS][1]
                P_fillterm=tetLHSdata[iLHS][2]
                Q_fillterm=tetLHSdata[iLHS][3]
            
                if elid in elid_val_P:
                    elid_val_P[elid] = elid_val_P[elid]+P_fillterm
                else:
                    elid_val_P[elid] = P_fillterm
                if elid in elid_val_Q:
                    elid_val_Q[elid] = elid_val_Q[elid]+Q_fillterm
                else:
                    elid_val_Q[elid] = Q_fillterm

        tetRHSdata=tet_info[itet][2]
        if len(tetRHSdata) > 0:
            for iRHS in range(len(tetRHSdata)):
                elid_rhs=tetRHSdata[iRHS][1]
                s1_fillterm=tetRHSdata[iRHS][2]
                if elid_rhs in elid_val_s1:
                    elid_val_s1[elid_rhs] = elid_val_s1[elid_rhs]+s1_fillterm
                else:
                    elid_val_s1[elid_rhs] = s1_fillterm

    row_id_s1 = list(elid_val_s1.keys())
    col_id_s1 = list(np.zeros((len(row_id_s1),), dtype = int))
    val_s1 = np.array(list(elid_val_s1.values()))    
    
    el_id_P = np.array(list(elid_val_P.keys()))
    val_P = np.array(list(elid_val_P.values()))

    el_id_Q = np.array(list(elid_val_Q.keys()))
    val_Q = np.array(list(elid_val_Q.values()))

    row_id_A, col_id_A = divmod(el_id_Q,n_in_edge) # P and Q have identical non zero element indices

    val_A = val_Q + 1j*omg*val_P
    A_coo = sp.coo_matrix((val_A, (row_id_A, col_id_A)), shape = (n_in_edge,n_in_edge))
    A_csc = A_coo.tocsc()

    val_b = -1j*omg*val_s1
    b_coo = sp.coo_matrix((val_b, (row_id_s1, col_id_s1)), shape = (n_in_edge,1))
    b_csc = b_coo.tocsc()
    
    toc1 = time.time()

    print('Set up ',n_in_edge,'X',n_in_edge,' FE Stiffness Matrix Vector Equation in ',toc1-tic1,' seconds.')

    print('solving FE stiffness matrix')

    FE_x = spsolve(A_csc,b_csc)

    toc2 = time.time()
    print() ; print ('SOLUTION TIME:', toc2-toc1, 'sec') ; print()

    # Post processing 

    dfrxpts=pd.read_csv(rxcoord_folder+rxcoord_filename+'.csv')
    reccoords=dfrxpts.to_numpy()

    recindx=list(range(len(reccoords)))
    B_recs = np.zeros((len(reccoords),7), dtype=float)
    for irx in recindx:
        Brec_i=postprocBrec(irx)
        B_recs[irx]=np.array(Brec_i)

    # E_recs = Parallel(n_jobs=job_cpus, prefer="processes")(delayed(Efield_postproc)(irx) for irx in recindx)
    # E_recs = Parallel(n_jobs=job_cpus, backend='loky')(delayed(Efield_postproc)(irx) for irx in recindx)

    toc3 = time.time()

    print() ; print ('POST PROCESSED B FIELD:', toc3-toc2, 'seconds.') ; print()

    xyzcolnames.extend(['real Bz','imag Bz','miw(real Bz)','miw(imag Bz)'])

    # exportdf=pd.DataFrame(np.array(E_recs),columns=xyzcolnames)
    exportdf=pd.DataFrame(B_recs,columns=xyzcolnames)
    exportdf.to_csv(export_filepath, index=False)
    
    toc4=time.time()
    print() ; print ('Done. Start to finish:', toc4-start_time, 'seconds.') ; print()