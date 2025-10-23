import sys
import pkg_resources


def check_versions():
    print("Python版本:", sys.version)
    print("\n已安装的库版本:")

    libraries = [
        'opencv-python', 'numpy', 'torch', 'torchvision', 'setuptools', 'pip',
        'cvzone', 'mediapipe', 'charset-normalizer'
    ]

    for lib in libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            print(f"{lib}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{lib}: 未安装")


if __name__ == "__main__":
    check_versions()