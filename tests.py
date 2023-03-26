from unittest import TestCase
from isciml import Mesh

class TestMesh(TestCase):

    def test_mesh_from_pyvista(self):
        fn = "test.vtk"
        mesh = Mesh(fn)
        npts, ncells, ndes, tet_ndes = mesh.load_mesh_files()
        assert(mesh.mesh.n_points == npts)
