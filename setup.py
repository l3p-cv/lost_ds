import setuptools
from pathlib import Path

requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setuptools.setup(
    name="lost_ds",
    version="donotchange",
    author="L3bm GmbH",
    author_email="info@l3bm.com",
    description="Lost Dataset library",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/l3p-cv/lost_ds",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
