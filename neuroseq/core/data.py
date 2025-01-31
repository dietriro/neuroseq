import csv
import inspect
import os.path

import numpy as np
import yaml
import datetime
from copy import copy
from os.path import exists

from neuroseq.common.config import *
from neuroseq.core.logging import log


COL_EXPERIMENT_ID = 0
COL_EXPERIMENT_MAP = 1
COL_EXPERIMENT_NUM = 2
COL_EXPERIMENT_MODE = 4


def load_yaml(path_yaml, file_name_yaml=None):
    if file_name_yaml is not None:
        path_yaml = join(path_yaml, file_name_yaml)
    with open(path_yaml) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            log.exception(exc)

    return data


def load_config(network_type, experiment_type=ExperimentType.EVAL_SINGLE, config_type=ConfigType.NETWORK,
                network_mode=None, map_name=None):
    if not inspect.isclass(network_type):
        network_type = type(network_type)

    if config_type == ConfigType.NETWORK:
        if experiment_type == ExperimentType.EVAL_SINGLE:
            plasticity_location = f"_{RuntimeConfig.plasticity_location}"
        else:
            plasticity_location = ""
        config_file_name = (f"{RuntimeConfig.config_prefix}_{experiment_type}_"
                            f"{RuntimeConfig.backend}_{network_type.__name__}{plasticity_location}.yaml")
    elif config_type == ConfigType.PLOTTING:
        config_file_name = f"{RuntimeConfig.config_prefix}_{config_type}"
        if network_mode is not None and os.path.exists(f"{config_file_name}_{network_mode.lower()}"):
            config_file_name += f"_{network_mode.lower()}"
            if map_name is not None and os.path.exists(f"{config_file_name}_{map_name.split('_')[0].lower()}"):
                    config_file_name += f"_{map_name.split('_')[0].lower()}"
        config_file_name += f".yaml"
    else:
        log.error(f"Unknown config type '{config_type}'. Aborting.")
        return None

    return load_yaml(RuntimeConfig.Paths.folders_config[config_type], config_file_name)


def get_last_experiment_num(experiment_id, experiment_type, experiment_map=None, network_mode=None) -> int:
    file_path = join(RuntimeConfig.Paths.folders_experiment[RuntimeConfig.backend], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    if not exists(file_path):
        return 0

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

    for line in lines[::-1]:
        if line[COL_EXPERIMENT_ID] == experiment_id:
            if experiment_map is None or line[COL_EXPERIMENT_MAP] == experiment_map:
                if network_mode is None or line[COL_EXPERIMENT_MODE] == network_mode:
                    return int(line[COL_EXPERIMENT_NUM])

    return 0


def get_last_instance(experiment_type, experiment_id, experiment_num, experiment_map=None):
    folder_path = get_experiment_folder(experiment_type, experiment_id, experiment_num, experiment_map=experiment_map)
    last_instance_id = 0
    for item in os.listdir(folder_path):
        if os.path.isdir(join(folder_path, item)):
            if item.isnumeric() and int(item) > last_instance_id:
                last_instance_id = int(item)

    return last_instance_id + 1


def get_experiment_folder(experiment_type, experiment_id, experiment_num, experiment_map=None,
                          experiment_subnum=None, instance_id=None):
    if experiment_map is None:
        folder_name = f"SHTMTotal_{experiment_id}_{experiment_num:02d}"
    else:
        folder_name = f"{experiment_id}_{experiment_map}_{experiment_num:02d}"
    folder_path = join(RuntimeConfig.Paths.folders_experiment[RuntimeConfig.backend],
                       str(EXPERIMENT_SUBFOLDERS[experiment_type]),
                       folder_name)
    folder_path_ret = folder_path

    if experiment_subnum is not None:
        folder_path_ret = join(folder_path, f"{experiment_subnum:0{RuntimeConfig.subnum_digits}d}")
        if not os.path.exists(folder_path_ret):
            log.debug(f"Folder '{folder_path}' does not exist. Trying different subnums {0, 1, ..., 5}.")
            for subnum_digits in range(5):
                folder_path_ret = join(folder_path, f"{experiment_subnum:0{subnum_digits}d}")
                if os.path.exists(folder_path_ret):
                    break
        if not os.path.exists(folder_path):
            log.error(f"Folder '{folder_path}' does not exist. Could not find any existing subnum.")
    if instance_id is not None:
        folder_path_ret = join(folder_path, f"{instance_id:0{RuntimeConfig.instance_digits}d}")

    return folder_path_ret


def get_experiment_file(file_type, experiment_path=None):
    if RuntimeConfig.file_prefix is not None and RuntimeConfig.file_prefix != "":
        prefix = f"{RuntimeConfig.file_prefix}_"
    else:
        prefix = ""
    file_name = f"{prefix}{file_type}"
    if experiment_path is not None:
        return join(experiment_path, file_name)
    else:
        return file_name


def save_setup(data, create_eval_file, do_update, file_path, experiment_id=None, experiment_num=None,
               experiment_map=None, network_mode=None, save_categories=False, max_decimals=5, **kwargs):
    # ToDo: Implement this feature, check if metrics can be added
    # add all static parameters defined above for this specific experiment
    # for param_name in sorted(kwargs):
    #     data[param_name] = kwargs[param_name]

    # check if new parameters were added since last time
    do_update_headers = False
    categories = []
    headers = []
    values = []
    lines = []
    prev_category = None
    categories_sparse = None
    if not create_eval_file:
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            lines = list(csv_reader)
        if save_categories:
            for category, header in zip(lines[0], lines[1]):
                if category == "":
                    if prev_category is None:
                        header_full = header
                    else:
                        header_full = f"{prev_category}.{header}"
                else:
                    header_full = f"{category}.{header}"
                    prev_category = category
                categories.append(category)
                headers.append(header)
                if header_full in data.keys():
                    values.append(data[header_full])
                    data.pop(header_full)
                else:
                    values.append('None')
        else:
            for header in lines[0]:
                headers.append(header)
                if header in data.keys():
                    values.append(data[header])
                    data.pop(header)
                else:
                    values.append('None')

        if len(data.keys()) > 0:
            do_update_headers = True

    if save_categories:
        for header_full, value in data.items():
            category = ".".join(header_full.split(".")[:-1])
            header = header_full.split(".")[-1]

            categories.append(category)
            headers.append(header)
            if max_decimals is not None and type(value) is float:
                values.append(np.round(value, max_decimals))
            else:
                values.append(value)

        categories_sparse = sparsen_list(categories)
    else:
        for header, value in data.items():
            headers.append(header)
            if max_decimals is not None and type(value) is float:
                values.append(np.round(value, max_decimals))
            else:
                values.append(value)

    start_id = 2 if save_categories else 1
    # writing to csv file
    with open(file_path, 'w' if create_eval_file or do_update_headers or do_update else 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the headers if file is newly created
        if create_eval_file or do_update_headers or do_update:
            if save_categories:
                csvwriter.writerow(categories_sparse)
            csvwriter.writerow(headers)

        # update last line if this is only an update
        if do_update and experiment_id is not None and experiment_num is not None and network_mode is not None:
            col_experiment_num = headers.index('experiment_num')
            col_network_mode = headers.index('network_mode')
            for line in lines[start_id:]:
                if int(line[COL_EXPERIMENT_NUM]) == experiment_num \
                        and line[COL_EXPERIMENT_MODE] == network_mode \
                        and line[COL_EXPERIMENT_ID] == network_mode \
                        and (experiment_map is None or line[COL_EXPERIMENT_MAP] == experiment_map):
                    line = values
                csvwriter.writerow(line)

        if do_update_headers:
            for line in lines[start_id:]:
                for header in data.keys():
                    line.append('None')
                csvwriter.writerow(line)

        # writing the data row
        if not do_update:
            csvwriter.writerow(values)

    return experiment_num


def save_instance_setup(parameters, performance, experiment_num=None, experiment_subnum=None, instance_id=None,
                        optimized_parameters=None, **kwargs):
    params = flatten_dict(parameters.dict(exclude_none=True))
    experiment_type = parameters.experiment.type
    experiment_id = parameters.experiment.id

    folder_path_instance = get_experiment_folder(experiment_type, experiment_id, experiment_num,
                                                 experiment_map=parameters.experiment.map_name,
                                                 experiment_subnum=experiment_subnum, instance_id=instance_id)
    if not os.path.exists(folder_path_instance):
        os.makedirs(folder_path_instance)

    folder_path_experiment = get_experiment_folder(experiment_type, experiment_id, experiment_num,
                                                   experiment_map=parameters.experiment.map_name,
                                                   experiment_subnum=None if instance_id is None else experiment_subnum,
                                                   instance_id=None)
    file_path = join(folder_path_experiment, EXPERIMENT_SETUP_FILE_NAME[ExperimentType.INSTANCE])
    create_eval_file = not exists(file_path)

    # prepare data
    if optimized_parameters is None:
        optimized_parameters = dict()
    data_end = {**optimized_parameters, **performance,
                **{'time_finished': datetime.datetime.now().strftime('%d.%m.%y - %H:%M')}}
    data_params = {name.lower().replace('.', '_'): params[name] for name in RuntimeConfig.saved_instance_params}
    data_exp = {'experiment_subnum': experiment_subnum} if instance_id is None else {'instance_id': instance_id}
    data = {**data_exp, **data_params, **data_end}

    save_setup(data, experiment_num, create_eval_file=create_eval_file, do_update=False, file_path=file_path,
               save_categories=False, **kwargs)

    return experiment_num


def save_experimental_setup(net, experiment_num=None, experiment_subnum=None, instance_id=None,
                            optimized_parameter_ranges=None, experiment_map=None, **kwargs):
    params = flatten_dict(net.p.dict(exclude_none=True))
    params.pop("config_type")
    experiment_type = net.p.experiment.type
    experiment_id = net.p.experiment.id

    file_path = join(RuntimeConfig.Paths.folders_experiment[RuntimeConfig.backend],
                     EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    # add the experiment id and network type
    create_eval_file = not exists(file_path)
    last_experiment_num = get_last_experiment_num(experiment_id, experiment_type, experiment_map,
                                                  network_mode=net.network_mode)
    do_update = False
    if experiment_num is None:
        experiment_num = last_experiment_num + 1
    elif experiment_num == last_experiment_num:
        do_update = True
    elif experiment_num < last_experiment_num:
        log.warning(f'Defined experiment num \'{experiment_num}\' is smaller than \'{last_experiment_num}\'. Aborting '
                    f'save process in order to prevent data loss.')
        return None

    # create folder if it doesn't exist
    folder_path = get_experiment_folder(experiment_type, experiment_id, experiment_num,
                                        experiment_map=experiment_map, experiment_subnum=experiment_subnum,
                                        instance_id=None)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # prepare data for saving
    if optimized_parameter_ranges is None:
        optimized_parameter_ranges = dict()
    data = {'experiment_id': experiment_id, 'experiment_map': experiment_map,
            'experiment_num': f'{experiment_num:02d}', 'network_type': str(net), 'network_mode': net.network_mode,
            'time_finished': datetime.datetime.now().strftime('%d.%m.%y - %H:%M')}

    data = {**data, **optimized_parameter_ranges, **params}

    save_setup(data, create_eval_file, do_update, file_path=file_path, save_categories=True,
               experiment_id=experiment_id, experiment_num=experiment_num, experiment_map=experiment_map,
               network_mode=net.network_mode, **kwargs)

    return experiment_num


def load_experimental_config(experiment_type: ExperimentType, network_type, experiment_num, experiment_id=None):
    """
    Loads the configuration of a previously performed experiment from a CSV file. The first line of the CSV file is
    expected to contain headers (metrics).
    :param experiment_type: The type of the experiment performed.
    :param network_type: The type of the network used for the experiment.
    :param experiment_num: The numeric index of the experiment for which the config shall be loaded.
    :param experiment_id: The alphabetic index of the experiment for which the config shall be loaded.
    :return: A list of all metrics or only of the metric given in the parameters of the function.
    """

    file_path = join(RuntimeConfig.Paths.folders_experiment[network_type.__name__], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    if not exists(file_path):
        return 0

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

    data = dict()
    headers = lines[0]
    csv_experiment_id = None
    csv_experiment_num = None
    for header_id, header in enumerate(headers):
        data[header] = None
        if header == 'experiment_id':
            csv_experiment_id = header_id
        elif header == 'experiment_num':
            csv_experiment_num = header_id

    data_found = False
    for line in lines[1:]:
        if (experiment_id is None or line[csv_experiment_id] == experiment_id) and \
                int(line[csv_experiment_num]) == experiment_num:
            data_found = True
            for header_id, header in enumerate(headers):
                data[header] = line[header_id]
            break

    if data_found:
        return data
    else:
        return None


def flatten_dict(original_dict, parent=None):
    new_dict = dict()
    for key, value in original_dict.items():
        if type(value) is dict:
            if parent is None:
                new_parent = key
            else:
                new_parent = f"{parent}.{key}"
            new_dict = {**new_dict, **flatten_dict(value, parent=new_parent)}
        else:
            if parent is None:
                new_dict[key] = value
            else:
                new_dict[f"{parent}.{key}"] = value
    return new_dict


def sparsen_list(list_full):
    list_sparse = list()
    prev_item = None
    for item in list_full:
        if prev_item is None or prev_item != item:
            list_sparse.append(item)
        else:
            list_sparse.append("")
        prev_item = copy(item)

    return list_sparse


def gen_map_name(map_name):
    if map_name is None:
        map_name = ""
        log.warning("Map name is None. Setting it to an empty string now.")
    return f"map_{map_name}"
