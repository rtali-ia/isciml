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
    def __init__(self, vtk_file_name: Union[str, os.PathLike]):
        if os.path.exists(vtk_file_name):
            self.vtk_file_name = vtk_file_name
        else:
            msg = "VTK file %s does not exist" % vtk_file_name
            log.error(msg)
            raise ValueError(msg)
        try:
            self.mesh = pv.read(self.vtk_file_name)
        except Exception as e:
            log.error(e)
            raise ValueError(e)

        self.npts = self.mesh.n_points
        self.ncells = self.mesh.n_cells
        self.nodes = np.array(self.mesh.points)
        self.tet_nodes = self.mesh.cell_connectivity.reshape((-1, 4))

    def get_centroids(self):
        nk = self.tet_nodes[:, 0]
        nl = self.tet_nodes[:, 1]
        nm = self.tet_nodes[:, 2]
        nn = self.tet_nodes[:, 3]
        self.centroids = (
            self.nodes[nk, :]
            + self.nodes[nl, :]
            + self.nodes[nm, :]
            + self.nodes[nn, :]
        ) / 4.0

    def get_volumes(self):
        ntt = len(self.tet_nodes)
        vot = np.zeros((ntt))
        for itet in np.arange(0, ntt):
            n1 = self.tet_nodes[itet, 0]
            n2 = self.tet_nodes[itet, 1]
            n3 = self.tet_nodes[itet, 2]
            n4 = self.tet_nodes[itet, 3]
            x1 = self.nodes[n1, 0]
            y1 = self.nodes[n1, 1]
            z1 = self.nodes[n1, 2]
            x2 = self.nodes[n2, 0]
            y2 = self.nodes[n2, 1]
            z2 = self.nodes[n2, 2]
            x3 = self.nodes[n3, 0]
            y3 = self.nodes[n3, 1]
            z3 = self.nodes[n3, 2]
            x4 = self.nodes[n4, 0]
            y4 = self.nodes[n4, 1]
            z4 = self.nodes[n4, 2]
            pv = (
                (x4 - x1) * ((y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1))
                + (y4 - y1) * ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1))
                + (z4 - z1) * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            )
            vot[itet] = np.abs(pv / 6.0)
        self.volumes = vot


class MagneticProperties:
    def __init__(
        self,
        file_name: Union[str, os.PathLike],
        kx: float = 1.0,
        ky: float = 1.0,
        kz: float = 1.0,
    ):
        if os.path.exists(file_name):
            self.file_name = file_name
        else:
            msg = "File %s does not exist" % file_name
            log.error(msg)
            raise ValueError(msg)

        try:
            self.properties = np.load(file_name)
        except Exception as e:
            log.error(e)
            raise ValueError(e)
        
        self.n_cells = self.properties.shape[0]
        if len(self.properties.shape) > 0:
            self.susceptibility = self.properties[:,0]
        else:
            self.susceptibility = self.properties
        
        if len(self.properties.shape) > 1:
            self.kx = self.properties[:,1]
        else:
            self.kx = np.full((self.n_cells,),kx)

        if len(self.properties.shape) > 2:
            self.ky = self.properties[:,2]
        else:
            self.ky = np.full((self.n_cells,),ky)

        if len(self.properties.shape) > 3:
            self.kz = self.properties[:,3]
        else:
            self.kz = np.full((self.n_cells,),kz)

        




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
