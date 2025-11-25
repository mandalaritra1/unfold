import subprocess
import sys

def install_package(package_name):
    # Install the package using subprocess to call pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# package_name = "pickle5"  # Replace with the package you want to install

# try:
#     import requests  # Replace with the module name of the package
# except ImportError:
#     print(f"{package_name} not found. Installing...")
#     install_package(package_name)
#     import requests  # Import again after installation

def import_packages():
    try:
        import mplhep  # Replace with the module name of the package
    except ImportError:
        print(f"{'mplhep'} not found. Installing...")
        install_package("mplhep")
        import mplhep  # Import again after installation
    try:
        import hist  # Replace with the module name of the package
    except ImportError:
        print(f"{'hist'} not found. Installing...")
        install_package("hist")
        import hist  # Import again after installation
    try:
        import pickle  # Replace with the module name of the package
    except ImportError:
        print(f"{'pickle5'} not found. Installing...")
        install_package("pickle5")
        import pickle  # Import again after installation
        
