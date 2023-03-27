from unittest import TestCase
from isciml import Mesh


class TestMesh(TestCase):
    def test_mesh_from_pyvista(self):
        fn = "test.vtk"
        mesh = Mesh(fn)
