import setuptools

setuptools.setup(
    name='audionmf',
    author="Tomáš Drbota",
    author_email="argoneuscze@gmail.com",
    description="A tool used for compressing audio using NMF.",

    long_description=open('README.md').read(),
    version='1.0.0',
    packages=setuptools.find_packages(exclude=['tests']),
    license='MIT',

    install_requires=[
        "click",
        "scipy>=1.2.0",
        "numpy",
        "dahuffman"
    ],

    setup_requires=["pytest-runner"],

    tests_require=["pytest"],

    extras_require={
        'dev': [
            "matplotlib"
        ]
    },

    entry_points={
        'console_scripts': [
            'audionmf = audionmf.__main__:main',
        ],
    }
)
