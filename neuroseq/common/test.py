import hashlib
import os.path

import yaml

from os.path import join
from abc import ABC

from neuroseq.core.logging import log
from neuroseq.core.data import load_yaml, get_experiment_folder, get_experiment_file, FileNames
from neuroseq.common.config import (RuntimeConfig, PY_PKG_PATH_DEFAULT, ExperimentType, Backends,
                                    NetworkMode, ConfigType)
from neuroseq.core.parameters import NetworkParameters

if RuntimeConfig.backend == Backends.NEST:
    from neuroseq.nest.network import SHTMTotal
elif RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    from neuroseq.brainscales2.network import SHTMTotal
else:
    raise Exception(f"Backend {RuntimeConfig.backend} not implemented yet. "
                    f"Please choose among [{Backends.BRAIN_SCALES_2}, {Backends.NEST}]")


DEFAULT_FILE_LIST = [
    FileNames.CONFIG[ConfigType.NETWORK],
    FileNames.EVENTS,
    FileNames.NETWORK,
    FileNames.PERFORMANCE,
    FileNames.PLASTICITY,
    FileNames.WEIGHTS
]


class Test(ABC):
    """
    A class for tests of neuroseq networks. Previously run simulations can be re-run with the functions of this class
    to verify the functionality of the current code state.
    """
    def __init__(self, experiment_num, experiment_map, experiment_id="test", experiment_type=ExperimentType.EVAL_SINGLE,
                 replay_target=None, file_list=None):

        self.experiment_num = experiment_num
        self.experiment_map = experiment_map
        self.experiment_id = experiment_id
        self.experiment_type = experiment_type

        self.replay_target = replay_target

        if file_list is None:
            self.file_list = DEFAULT_FILE_LIST

        self.test_types = {
            "predictive": self.predictive,
            "replay": self.replay
        }

        self.pkg_path_tmp = "/tmp/neuroseq"

        self.network: SHTMTotal = None

    def predictive(self):
        """
        Runs an experiment in predictive mode.
        """
        log.info(f"\nRunning predictive test with `experiment_id={self.experiment_id}`, "
                 f"`experiment_num={self.experiment_num}`, and `experiment_map={self.experiment_map}`.\n")

        # Load config file from original test
        p = NetworkParameters(network_type=SHTMTotal)
        p.load_experiment_params(experiment_type=self.experiment_type,
                                 experiment_id=self.experiment_id,
                                 experiment_num=self.experiment_num,
                                 experiment_map=self.experiment_map)

        self.network = SHTMTotal(p=p, experiment_num=self.experiment_num)
        self.network.init_backend(offset=0)

        self.network.init_plasticity_rule()
        self.network.init_neurons()
        self.network.init_connections()

        self.network.init_external_input()
        self.network.init_rec_exc()
        self.network.init_prerun()

        # Run simulation of new test
        self.network.run()

    def replay(self):
        """
        Runs an experiment in replay mode.
        """
        log.info(f"\nRunning replay test with `experiment_id={self.experiment_id}`, "
                 f"`experiment_num={self.experiment_num}`, and `experiment_map={self.experiment_map}`.\n")

        # Reset network state for prediction
        RuntimeConfig.file_prefix = NetworkMode.PREDICTIVE

        # Run prediction to learn sequences first
        self.predictive()

        # Reset network state for replay
        RuntimeConfig.file_prefix = NetworkMode.REPLAY

        # Load config file from original replay test
        p_replay = NetworkParameters(network_type=SHTMTotal)
        p_replay.load_experiment_params(experiment_type=self.experiment_type,
                                        experiment_id=self.experiment_id,
                                        experiment_num=self.experiment_num,
                                        experiment_map=self.experiment_map)

        # Set the network state to replay
        self.network.set_state(NetworkMode.REPLAY, target=p_replay.replay.target)

        # Run replay
        self.network.run(steps=p_replay.experiment.episodes, plasticity_enabled=False,
                         runtime=p_replay.experiment.runtime)

    def save_network_data(self):
        """
        Saves the entire network data to a temporary location for later analysis and comparison to existing tests.
        """
        # Create and set temporary package path
        os.makedirs(join(self.pkg_path_tmp, "data"), exist_ok=True)
        # RuntimeConfig.pkg_path = self.pkg_path_tmp
        RuntimeConfig.Paths.update_package_path(self.pkg_path_tmp)

        # Save the data from the experiment there
        self.network.save_full_state()

    def run_test(self, test_type, generate_missing_checksums=False):
        """
        Runs a test with the given test type including:

        - checksum loading from existing test
        - parameter loading from existing files
        - simulation/emulation of the defined test network
        - data saving of network information
        - comparison of new data to existing test

        :param test_type: The type of the test to be evaluated: ["predictive", "replay"]
        :type test_type: str
        :param generate_missing_checksums: Whether to generate missing checksum file using the `DEFAULT_FILE_LIST`.
        :type generate_missing_checksums: bool
        :return: Nothing
        :rtype: None
        """
        # Reset pkg path to default
        RuntimeConfig.Paths.update_package_path(PY_PKG_PATH_DEFAULT)
        RuntimeConfig.file_prefix = test_type

        # Get checksums from original test
        checksums_original = self.load_checksums(experiment_id=self.experiment_id,
                                                 experiment_num=self.experiment_num,
                                                 experiment_map=self.experiment_map,
                                                 experiment_type=self.experiment_type,
                                                 generate_missing_checksums=generate_missing_checksums)
        if checksums_original is None:
            log.error("Checksums not loaded successfully. Cannot proceed. Terminating now.")
            return

        # Run experiment
        self.test_types[test_type]()

        # Save results
        self.save_network_data()

        # Check results
        test_exp_path = get_experiment_folder(self.experiment_type, self.experiment_id, self.experiment_num,
                                              experiment_map=self.experiment_map)
        self.check_files(file_list=self.file_list, path=test_exp_path, checksums_original=checksums_original)

    @staticmethod
    def check_files(path, checksums_original, file_list=None):
        """
        Compares the md5sums of the files in `file_list` located at `path` with the original ones contained in
        `checksums_original`.

        :param path: The path to the experiment folder where the files are located.
        :type path: str
        :param checksums_original: The original checksums with names as keys and md5sums as values.
        :type checksums_original: dict
        :param file_list: The
        :type file_list: list(str)
        :return: The number of correctly evaluated checksums (files).
        :rtype: int
        """
        log.info("\nChecksum test results:")

        if file_list is None:
            log.warning("No file list specified, using `DEFAULT_FILE_LIST`.")
            file_list = DEFAULT_FILE_LIST

        num_checksums_correct = 0
        for file_name in file_list:
            file_path = get_experiment_file(file_name, experiment_path=path)
            checksum_new = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            num_checksums_correct += Test.checksum_test(file_name, checksums_original[file_name], checksum_new)

        if num_checksums_correct == len(file_list):
            log.info(f"SUCCESS: All {num_checksums_correct} checks have successfully passed.")
        else:
            log.warning(f"WARNING: {len(file_list)-num_checksums_correct} check(s) have not been passed successfully.")

        return num_checksums_correct

    @staticmethod
    def checksum_test(file_name, checksum_original, checksum_new):
        """
        Runs a checksum test of the md5sum strings in `checksum_original` and `checksum_new`.

        :param file_name: The name of the file, including the type.
        :type file_name: str
        :param checksum_original: The checksum of the original file.
        :type checksum_original: str
        :param checksum_new: The checksum of the new file.
        :type checksum_new: str
        :return: True, if check revealed that the checksums are equal, else False.
        :rtype: bool
        """
        if checksum_original == checksum_new:
            result = "OK"
        else:
            result = "FAILED"

        log.info(f"\t{file_name}: {result}")

        return result == "OK"

    @staticmethod
    def generate_checksums(experiment_num, experiment_map, experiment_id="test",
                           experiment_type=ExperimentType.EVAL_SINGLE, network_mode=None, file_list=None):
        """
        Generates md5sums for the experiment defined by the parameters below.

        :param experiment_num: The running number of the experiment with the given id/map.
        :type experiment_num: int
        :param experiment_map: The map used for the experiment.
        :type experiment_map: str
        :param experiment_id: The identifier of the network, e.g. "test" or "eval".
        :type experiment_id: str
        :param experiment_type: The type of the experiment, e.g. `ExperimentType.EVAL_SINGLE`.
        :type experiment_type: str
        :param network_mode: The mode of the network during the experiment ("predictive" or "replay").
        :type network_mode: str
        :param file_list: The list of files for which a checksum should be generated.
        :type file_list: list(str)
        :return: The checksums in a dictionary format (file_name: checksum) if successful, None if not.
        :rtype: dict
        """
        if file_list is None:
            file_list = DEFAULT_FILE_LIST

        if network_mode is not None:
            RuntimeConfig.file_prefix = network_mode

        experiment_path = get_experiment_folder(experiment_type, experiment_id, experiment_num,
                                                experiment_map=experiment_map)

        # Generate checksums for all files in file_list
        checksums = dict()
        for file_name in file_list:
            file_path = get_experiment_file(file_name, experiment_path=experiment_path)
            if not os.path.exists(file_path):
                log.warning(f"Couldn't find file at: {file_path}")
                continue
            checksums[file_name] = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        file_check_path = get_experiment_file(FileNames.TEST, experiment_path=experiment_path)
        with open(file_check_path, 'w') as file_check:
            yaml.dump(checksums, file_check)

    @staticmethod
    def load_checksums(experiment_num, experiment_map=None, experiment_id="test",
                       experiment_type=ExperimentType.EVAL_SINGLE, experiment_subnum=None, instance_id=None,
                       generate_missing_checksums=True):
        """
        Load md5sums from an existing file and possibly generate them using the `DEFAULT_FILE_LIST`
        if the file doesn't exist.

        :param experiment_num: The running number of the experiment with the given id/map.
        :type experiment_num: int
        :param experiment_map: The map used for the experiment.
        :type experiment_map: str
        :param experiment_id: The identifier of the network, e.g. "test" or "eval".
        :type experiment_id: str
        :param experiment_type: The type of the experiment, e.g. `ExperimentType.EVAL_SINGLE`.
        :type experiment_type: str
        :param experiment_subnum: The sub-number of the experiment, if experiment_type == `ExperimentType.OPT_GRID/OPT_GRID_MULTI`.
        :type experiment_subnum: int
        :param instance_id: The instance id of the experiment, if experiment_type == `ExperimentType.EVAL_MULTI/OPT_GRID_MULTI`
        :type instance_id: int
        :param generate_missing_checksums: Whether to generate missing checksum file using the `DEFAULT_FILE_LIST`.
        :type generate_missing_checksums: bool
        :return: The md5sums loaded from file in a dictionary format (file_name: checksum).
        :rtype: dict
        """
        folder_path = get_experiment_folder(experiment_type, experiment_id, experiment_num,
                                            experiment_map=experiment_map, experiment_subnum=experiment_subnum,
                                            instance_id=instance_id)
        checkfile_path = get_experiment_file(FileNames.TEST, experiment_path=folder_path)


        if not os.path.exists(checkfile_path):
            log.warning(f"Couldn't find checkfile at: {checkfile_path}")
            if generate_missing_checksums:
                log.info(f"Generating default checkfile at: {checkfile_path}")
                Test.generate_checksums(experiment_num, experiment_map, experiment_id=experiment_id,
                                        experiment_type=experiment_type)
            else:
                log.warning("Aborting checksum loading.")
                return None

        checksums = load_yaml(path_yaml=checkfile_path)

        return checksums
