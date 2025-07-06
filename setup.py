from setuptools import setup, find_packages

setup(
    name="ap-bsn",
    version="0.1.0",
    description="AP-BSN: Attention-Powered Blind Spot Networks for Image Denoising",
    author="AP-BSN Team",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "scikit-image",
        "scipy",
    ],
    python_requires=">=3.8",
) 