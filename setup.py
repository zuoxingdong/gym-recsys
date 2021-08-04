from setuptools import setup


setup(
    name='gym_recsys',
    author='Xingdong Zuo',
    version='0.0.1',
    keywords='recsys, rl, reinforcement-learning, openai-gym, gym, python',
    url='https://github.com/zuoxingdong/gym-recsys',
    description='Customizable RecSys Simulator for OpenAI Gym',
    packages=['gym_recsys', 'gym_recsys.envs'],
    install_requires=[
        'gym>=0.17.3',
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'seaborn>=0.11.1',
        'matplotlib>=3.4.0',
        'Pillow>=8.1.0'
    ],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
