import click
import numpy as np
import pandas as pd
import logging
from rich.logging import RichHandler
from tqdm import tqdm
import sys
import yaml
import os
import pyvista as pv
import calc_mig_all
from typing import Union
from mpi4py import MPI
import ctypes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format="Rank: " + str(rank) + "/" + str(size) + ": %(asctime)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
log = logging.getLogger("rich")


class Mesh:
    def __init__(self, vtk_file_name: Union[str, os.PathLike]):
        log.debug("Reading vtk file %s" % vtk_file_name)
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
        log.debug("Reading mesh completed")

        self.npts = self.mesh.n_points
        self.ncells = self.mesh.n_cells
        self.nodes = np.array(self.mesh.points)
        self.tet_nodes = self.mesh.cell_connectivity.reshape((-1, 4))
        log.debug("Generated mesh properties")

    def __str__(self):
        return str(self.mesh)

    def get_centroids(self):
        log.debug("Getting centroids")
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
        log.debug("Getting centroids done!")

    def get_volumes(self):
        log.debug("Getting volumes")
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
        log.debug("Getting volumes done!")


class MagneticProperties:
    def __init__(
        self,
        file_name: Union[str, os.PathLike],
        kx: float = 1.0,
        ky: float = 1.0,
        kz: float = 1.0,
    ):
        log.debug("Reading magnetic properties %s" % file_name)
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
        log.debug("Reading magnetic properties %s done!" % file_name)

        if len(self.properties.shape) > 0:
            self.n_cells = self.properties.shape[0]
        else:
            msg = "Magnetic properties file %s is incorrect" % file_name
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

        log.debug("Setting all magnetic properties done!")


class MagneticAdjointSolver:
    def __init__(
        self,
        reciever_file_name: Union[str, os.PathLike],
        Bx: float = 4594.8,
        By: float = 19887.1,
        Bz: float = 41568.2,
    ):
        log.debug("Solver initialization started!")
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
        log.debug("Solver initialization done!")

    def solve(self, mesh: Mesh, magnetic_properties: MagneticProperties):
        log.debug("Solver started for %s" % magnetic_properties.file_name)
        rho_sus = np.zeros((1000000), dtype=float)
        rho_sus[0 : mesh.ncells] = magnetic_properties.susceptibility

        ctet = np.zeros((1000000, 3), dtype=float)
        ctet[0 : mesh.ncells] = np.float32(mesh.centroids)

        vtet = np.zeros((1000000), dtype=float)
        vtet[0 : mesh.ncells] = np.float32(mesh.volumes)

        nodes = np.zeros((1000000, 3), dtype=float)
        nodes[0 : mesh.npts] = np.float32(mesh.nodes)

        tets = np.zeros((1000000, 4), dtype=int)
        tets[0 : mesh.ncells] = mesh.tet_nodes + 1

        n_obs = len(self.receiver_locations)
        rx_loc = self.receiver_locations.to_numpy()

        obs_pts = np.zeros((1000000, 3), dtype=float)
        obs_pts[0:n_obs] = np.float32(rx_loc[:, 0:3])

        ismag = True
        rho_sus = rho_sus * self.Bv

        istensor = False
        mig_data = calc_mig_all.calc_and_mig_all_rx(
            rho_sus,
            ismag,  # this calls a function calc_and_mig_field
            istensor,  # We input all the arrays required
            self.LX,
            self.LY,
            self.LZ,
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
        log.debug("Solver done for %s" % magnetic_properties.file_name)
        return mig_data[0 : mesh.ncells]


@click.command()
@click.option(
    "--config_file",
    help="Configuration file in YAML format",
    type=click.Path(),
    required=True,
    show_default=True,
)
def isciml(config_file: os.PathLike):
    log.debug("Reading configuration file")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as fp:
                config = yaml.safe_load(fp)
        except Exception as e:
            log.error(e)
            raise ValueError(e)
    else:
        msg = "File %s doesn't exist" % config_file
        log.error(msg)
        raise ValueError(msg)

    log.debug("Reading configuration file done!")

    mesh = Mesh(config["vtk_file"])
    mesh.get_centroids()
    mesh.get_volumes()

    properties = MagneticProperties(config["magnetic_properties_file"])
    solver = MagneticAdjointSolver(config["receiver_locations_file"])
    output = solver.solve(mesh, properties)
    # np.save("output.npy", output)
    return 0


if __name__ == "__main__":
    sys.exit(isciml())  # pragma: no cover
