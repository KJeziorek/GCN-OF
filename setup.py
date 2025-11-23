from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='matrix_neighbour',
    ext_modules=[
        CppExtension(
            name='matrix_neighbour',
            sources=['dataset/utils/matrix_neighbour.cpp'],
            extra_compile_args=['-O3']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
