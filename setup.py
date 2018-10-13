import setuptools

setuptools.setup(
    name='audionmf',
    author="Tomáš Drbota",
    author_email="argoneuscze@gmail.com",
    description="A tool used for compressing audio using NMF.",

    long_description=open('README').read(),
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='MIT',

    install_requires=[
        "click",
        "numpy"
    ],

    entry_points={
        'console_scripts': [
            'audionmf = audionmf.__main__:main',
        ],
    }
)
