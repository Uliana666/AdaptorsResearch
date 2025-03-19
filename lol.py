from packaging.version import parse

def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    packages = {}
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        pkg_name, version = line.strip().split('==')
        if pkg_name in packages:
            if parse(version) > parse(packages[pkg_name]):
                packages[pkg_name] = version
        else:
            packages[pkg_name] = version
    return packages

def write_requirements(packages, filename='updated_req.txt'):
    with open(filename, 'w') as file:
        for pkg, version in packages.items():
            file.write(f"{pkg}=={version}\n")

if __name__ == "__main__":
    requirements_file = 'req.txt'
    packages = parse_requirements(requirements_file)
    write_requirements(packages, 'updated_req.txt')
