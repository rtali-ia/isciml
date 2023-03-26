import click
import numpy as np
import logging
from rich.logging import RichHandler
from tqdm import tqdm
from pydantic import BaseModel
import sys
import pyaml 
import os
import pyvista as pv
from typing import Union

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

class Mesh:

    def __init__(self,vtk_file_name: Union[str, os.PathLike]):
        if os.path.exists(vtk_file_name):
            self.vtk_file_name = vtk_file_name
        else:
            msg = "VTK file %s does not exist"%vtk_file_name
            log.error(msg)
            raise ValueError(msg)
        try:
            self.mesh = pv.read(self.vtk_file_name)
        except Exception as e:
            raise ValueError(e)

    def load_mesh_files(self):
        txtopen = open(self.vtk_file_name,'r')
        all_vtk_lines = txtopen.readlines()
        txtopen.close()
        skp_hdr_ln_no = 3 # Line no. 4
        n_pts_ln_no = skp_hdr_ln_no + 1 # This line contains information about the number of nodes
        npts = int(all_vtk_lines[n_pts_ln_no].split()[1]) # The number of nodes in the mesh

        ndes = np.zeros((npts,3), dtype='float')

        for i_pt in range(0, npts):
            ndes[i_pt,:] = list(map(float, all_vtk_lines[n_pts_ln_no+1+i_pt].split()[0:])) # Read in the coordinates of the nodes
        
        cells_ln_no = n_pts_ln_no + 1 + npts + 1 # This line contains information about the number of tetrahedral cells.
        ncells = int(all_vtk_lines[cells_ln_no].split()[1]) # Get the value of the number of cells (tetrahedra) in the mesh

        tet_ndes = np.zeros((ncells, 4), dtype='int')

        for i_cell in range(0, ncells):
            tet_ndes[i_cell,:] = list(map(int, all_vtk_lines[cells_ln_no+1+i_cell].split()[1:])) # Read in the node indices (starting from 0) for individual tetrahedron
            
        return npts, ncells, ndes, tet_ndes

@click.command()
@click.option(
    "--config",
    help="Configuration file in YAML format",
    required=True,
    show_default=True,
)
def isciml(**kwargs):
    log.info(kwargs)
    return 0

if __name__ == "__main__":
    sys.exit(isciml())  # pragma: no cover
