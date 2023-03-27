from setuptools import setup

setup(
    name="isciml",
    version="0.1.0",
    author='Santi Adavani',
    author_email='santis@gmail.com',
    url='https://github.com/santiadavani/isciml',
    py_modules=["isciml"],
    install_requires=[
        "click==8.1.3",
        "mpi4py==3.1.4",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "black==23.1.0",
        "pyaml==21.10.1",
        "rich==13.3.2",
        "tqdm==4.65.0",
        "pyvista==0.38.5",
    ],
    entry_points={
        "console_scripts": [
            "isciml = isciml:isciml",
        ],
    },
)
