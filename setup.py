import os
import sys
import subprocess
import setuptools
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path

class CMakeExtension(Extension):
    """Defines a CMake extension for compiling C++ code."""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

def ensure_pybind11():
    pass
#     try:
#         import pybind11
#         print(f"Pybind11 already installed: {pybind11.__version__}")
#     except ImportError:
#         print("Installing pybind11 before compilation...")
#         print(sys.executable)
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])

class CMakeBuild(build_ext):
    """Runs CMake and Make to build the shared library."""
    def build_extension(self, ext):
        ensure_pybind11()
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        python_include_dir = sysconfig.get_path("include")
        python_library_dir = sysconfig.get_config_var("LIBDIR")
        
        # MODIFIED 19/05 DOCKER TESTS vv
        # python_lib_name = sysconfig.get_config_var('INSTSONAME') or \
        #              f"libpython{sysconfig.get_config_var('VERSION')}.so"
        # python_library_dir = os.path.join(
        #     sysconfig.get_config_var('LIBDIR'),
        #     python_lib_name
        # )
        ###


        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_INCLUDE_DIR={python_include_dir}",
            f"-DPython3_LIBRARY_DIR={python_library_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_MAKE_PROGRAM=make",
            f"-DPython3_ROOT_DIR={os.path.dirname(os.path.dirname(sys.executable))}",
        ]


        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        try:
            # Run CMake configuration
            subprocess.run(["cmake", ext.sourcedir] + cmake_args, 
                        cwd=build_temp, 
                        check=True,
                        capture_output=True,  # This captures the output
                        text=True)
            
            # Compile the C++ extension
            subprocess.run(["cmake", "--build", ".", "--config", "Release"],
                        cwd=build_temp,
                        check=True,
                        capture_output=True,  # This captures the output
                        text=True)
        
        except subprocess.CalledProcessError as e:
            print(f"CMake configuration failed with output:\n{e.stdout}\n{e.stderr}")
            raise
        # Explicitly copy the built library to the expected location
        lib_name = f"{ext.name}{sysconfig.get_config_var('EXT_SUFFIX')}"
        src_path = os.path.join(extdir, lib_name)
        dest_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if os.path.exists(src_path):
            self.copy_file(src_path, dest_path)

class CustomInstall(install):
    """Ensures the C++ extension is built during pip install."""
    def run(self):
        self.run_command("build_ext")
        install.run(self)

class CustomDevelop(develop):
    """Ensures the C++ extension is built during development install."""
    def run(self):
        self.run_command("build_ext")
        develop.run(self)

# Recursively collect all files in myfolder
# data_files = []
# for root, _, files in os.walk("c_implementation"):
#     for file in files:
#         data_files.append(os.path.join(root, file))

data_files = [str(file.relative_to("uaibot")) for file in Path("uaibot/c_implementation").rglob("*") if file.is_file()]

#Setup the installation
setup(
    name="uaibot",
    version="1.2.2",
    author="Vinicius Goncalves et al",
    author_email="vinicius.marianog@gmail.com",
    description="Uaibot, online robotic simulator",
    long_description="Uaibot, online robotic simulator",
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your_repo",
    #packages=setuptools.find_packages(where=".",include=["uaibot", "uaibot.*"]),
    packages=["uaibot"] + setuptools.find_packages(include=["uaibot.*"]),
    package_dir={"uaibot": "uaibot"},  
    include_package_data=True,  
    ext_modules=[CMakeExtension("uaibot_cpp_bind", sourcedir="uaibot/c_implementation")],
    cmdclass={
        "build_ext": CMakeBuild,
        "install": CustomInstall,
        "develop": CustomDevelop,
    },
    # package_data={
    #     "simulation": ["**/*.js"], 
    # },

    package_data={
        "uaibot.simulation": ["*.js"], 
        "uaibot": data_files
    },
     
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "pybind11>=2.10",
        "colour>=0.1.5",
        "httplib2>=0.20.4",
        "ipython>=8.0.1",
        "numpy>=1.24", 
        "scipy>=1.10",  
        "quadprog>=0.1.13",
        "matplotlib >= 3.10.0",
        "requests"
    ],
    setup_requires=[
        "pybind11>=2.10",
        # "setuptools>=58.0.4",
        ],
)
