import argparse

from os.path import join
import subprocess

from neuroseq.common.config import NetworkMode, RuntimeConfig, Backends, NetworkMode
from neuroseq.core.logging import log
from neuroseq.core.data import load_yaml


def parse_arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Run a test of a neuroseq network by specifying the `backend` used'
                                                 'for simulation/emulation and the `mode` of the network.')

    # Required positional argument
    parser.add_argument('backend', type=str,
                        help='The backend used for the test experiment: [`nest`, `bss2`]')

    # Optional positional argument
    parser.add_argument('network_mode', type=str,
                        help='The mode of the test network: [`predictive`, `replay`]')

    # Switch
    parser.add_argument('-i', '--index', action='store', type=int, dest="test_index", default=0,
                        help='The index of the test case that should be used, default is 0.')

    # Switch
    parser.add_argument('-g', '--generate-missing-checksums',  action='store_true',
                        dest="generate_missing_checksums",
                        help='If set, missing checksum files will be generated using default values.')

    # Switch
    parser.add_argument('-d', '--docker', action='store_true',
                        dest="docker",
                        help='If set, the test is run in an isolated docker container.')

    args = parser.parse_args()

    if args.backend not in Backends.get_all():
        log.error(f"Specified backend does not exist: {args.backend}. Terminating.")
        exit(0)

    if args.network_mode not in NetworkMode.get_all():
        log.error(f"Specified network mode does not exist: {args.network_mode}. Terminating.")
        exit(0)

    return args


def run_test_local(backend, network_mode, test_index=0, generate_missing_checksums=False):

    from neuroseq.common.test import Test

    # Load config from file
    config = load_yaml(join(RuntimeConfig.Paths.package, 'config', 'tests_shtm.yaml'))
    config = config[backend][network_mode][test_index]

    # Initialize and run test
    test = Test(experiment_id=config["experiment_id"], experiment_num=config["experiment_num"],
                experiment_map=config["experiment_map"])

    test.run_test(network_mode, generate_missing_checksums=generate_missing_checksums)


def run_test_docker(backend, network_mode, test_index=0, generate_missing_checksums=False):
    docker_backends = {Backends.NEST: 'pynn-nest', Backends.BRAIN_SCALES_2: 'bss2'}

    add_params = "-g" if generate_missing_checksums else ""

    # Initialize and run test
    subprocess.run(f'docker run -it --rm --name neuroseq:{docker_backends[backend]} '
                   f'python scripts/test_run.py {backend} {network_mode} -i {test_index} {add_params}')


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Set backend according to arguments
    RuntimeConfig.backend = args.backend

    # Run test
    if args.docker:
        run_test_docker(args.backend, args.network_mode, test_index=args.test_index,
                        generate_missing_checksums=args.generate_missing_checksums)
    else:
        run_test_local(args.backend, args.network_mode, test_index=args.test_index,
                       generate_missing_checksums=args.generate_missing_checksums)
