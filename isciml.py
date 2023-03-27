import click
import numpy as np
import pandas as pd
import logging
from rich.logging import RichHandler
from tqdm import tqdm
from pydantic import BaseModel
import sys
import pyaml
import os
import pyvista as pv
import calc_and_mig_kx_ky_kz
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

    def __str__(self):
        return str(self.mesh)

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

        if len(self.properties.shape) > 0:
            self.n_cells = self.properties.shape[0]
        else:
            msg = "Magnetic properties file %s is incorrect"%file_name
            log.error(msg)
            raise ValueError(msg)
        
        if self.properties.ndim == 1:
            self.properties = np.expand_dims(self.properties, axis=1)

        if self.properties.shape[1] > 0:
            self.susceptibility = self.properties[:, 0]

        if self.properties.shape[1] > 1:
            self.kx = self.properties[:, 1]
        else:
            self.kx = np.full((self.n_cells,), kx)

        if self.properties.shape[1] > 2:
            self.ky = self.properties[:, 2]
        else:
            self.ky = np.full((self.n_cells,), ky)

        if self.properties.shape[1] > 3:
            self.kz = self.properties[:, 3]
        else:
            self.kz = np.full((self.n_cells,), kz)


class MagneticAdjointSolver:
    def __init__(
        self,
        reciever_file_name: Union[str, os.PathLike],
        Bx: float = 4594.8,
        By: float = 19887.1,
        Bz: float = 41568.2,
    ):
        if os.path.exists(reciever_file_name):
            self.reciever_file_name = reciever_file_name
        else:
            msg = "File %s does not exist" % reciever_file_name
            log.error(msg)
            raise ValueError(msg)

        try:
            self.receiver_locations = pd.read_csv(reciever_file_name)
        except Exception as e:
            log.error(e)
            raise ValueError(e)

        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Bv = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
        self.LX = np.float32(self.Bx / self.Bv)
        self.LY = np.float32(self.By / self.Bv)
        self.LZ = np.float32(self.Bz / self.Bv)

    def solve(self, mesh: Mesh, magnetic_properties: MagneticProperties):
        rho_sus = np.zeros((10000000), dtype="float32")
        rho_sus[0 : mesh.ncells] = magnetic_properties.susceptibility

        KXt = np.zeros((10000000), dtype="float32")
        KXt[0 : mesh.ncells] = magnetic_properties.kx

        KYt = np.zeros((10000000), dtype="float32")
        KYt[0 : mesh.ncells] = magnetic_properties.ky

        KZt = np.zeros((10000000), dtype="float32")
        KZt[0 : mesh.ncells] = magnetic_properties.kz

        ctet = np.zeros((10000000, 3), dtype="float32")
        ctet[0:mesh.ncells] = np.float32(mesh.centroids)

        vtet = np.zeros((10000000), dtype="float32")
        vtet[0:mesh.ncells] = np.float32(mesh.volumes)

        nodes = np.zeros((10000000, 3), dtype="float32")
        nodes[0 : mesh.npts] = np.float32(mesh.npts)

        tets = np.zeros((10000000, 4), dtype=int)
        tets[0 : mesh.ncells] = mesh.tet_nodes + 1

        n_obs = len(self.receiver_locations)
        rx_loc = self.receiver_locations.to_numpy()

        obs_pts = np.zeros((1000000, 3), dtype="float32")
        obs_pts[0:n_obs] = np.float32(rx_loc[:, 0:3])
        
        log.info(rx_loc)

        ismag = True
        rho_sus = rho_sus * self.Bv

        istensor = False

        mig_data = calc_and_mig_kx_ky_kz.calc_and_mig_field(
            rho_sus,
            ismag,
            istensor,
            KXt,
            KYt,
            KZt,
            self.LX,
            self.LY,
            self.LZ,
            nodes,
            tets,
            mesh.ncells,
            obs_pts,
            n_obs,
            ctet,
            vtet,
        )
        return mig_data[0:ntets]



@click.command()
@click.option(
    "--config",
    help="Configuration file in YAML format",
    required=True,
    show_default=True,
)
def isciml(**kwargs):
    log.info(kwargs)
    
    mesh = Mesh("test.vtk")
    mesh.get_centroids()
    mesh.get_volumes()
    
    properties = MagneticProperties("material_properties.npy")
    solver = MagneticAdjointSolver("receiver_locations.csv")
    output = solver.solve(mesh, properties)
    
    return 0


if __name__ == "__main__":
    sys.exit(isciml())  # pragma: no cover
