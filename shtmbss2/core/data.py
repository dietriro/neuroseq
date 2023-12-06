import csv
import inspect
import numpy as np
import yaml
import datetime
from copy import copy
from os.path import exists

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
# from shtmbss2.common.network import SHTMBase


def load_yaml(path_yaml, file_name_yaml):
    with open(join(path_yaml, file_name_yaml)) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            log.exception(exc)

    return data


def load_config(network_type):
    if not inspect.isclass(network_type):
        network_type = type(network_type)

    config_file_name = f"{RuntimeConfig.config_prefix}_{network_type.__name__}.yaml"
    return load_yaml(PATH_CONFIG, config_file_name)


def get_last_experiment_num(net, experiment_id, experiment_type) -> int:
    file_path = join(EXPERIMENT_FOLDERS[RuntimeConfig.backend], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    if not exists(file_path):
        return 0

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

    for line in lines[::-1]:
        if line[0] == experiment_id:
            return int(line[1])

    return 0


def get_experiment_folder(net, experiment_id, experiment_num, instance_id=None):
    folder_path = join(EXPERIMENT_FOLDERS[RuntimeConfig.backend], f"{str(net)}_{experiment_id}_{experiment_num:02d}")
    if instance_id is not None:
        folder_path = join(folder_path, f'{instance_id:02d}')
    return folder_path


def save_setup(data, experiment_num, create_eval_file, do_update, file_path, save_categories=False, **kwargs):

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
            values.append(value)

        categories_sparse = sparsen_list(categories)
    else:
        for header, value in data.items():
            headers.append(header)
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
        if do_update:
            experiment_num_col = headers.index('experiment_num')
            for line in lines[start_id:]:
                if int(line[experiment_num_col]) == experiment_num:
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


def save_config(net, instance_id=None):
    experiment_id = net.p.Experiment.id
    experiment_num = net.experiment_num

    folder_path = get_experiment_folder(net, experiment_id, experiment_num, instance_id=instance_id)
    file_path = join(folder_path, f"config.yaml")

    with open(file_path, 'w') as file:
        yaml.dump(net.p.dict(exclude_none=True), file)


def save_performance_data(data, metric_names, net, experiment_num, instance_id=None):
    experiment_id = net.p.Experiment.id

    folder_path = get_experiment_folder(net, experiment_id, experiment_num, instance_id=instance_id)

    for i_metric, metric in enumerate(data):
        file_path = join(folder_path,
                         f"{metric_names[i_metric]}")

        np.save(file_path, metric)


def save_network_data(net, experiment_num, instance_id=None):
    experiment_id = net.p.Experiment.id

    folder_path = get_experiment_folder(net, experiment_id, experiment_num, instance_id=instance_id)
    file_path = join(folder_path, "weights")

    weights_dict = {var_name: getattr(net, var_name) for var_name in RuntimeConfig.saved_network_vars}
    for con_name, connections in weights_dict.items():
        weights_all = list()
        for connection in connections:
            weights_all.append(connection.get("weight", format="array"))
        weights_dict[con_name] = np.array(weights_all)

    np.savez(file_path, **weights_dict)

    file_path = join(folder_path, "plasticity")

    plasticity_dict = {var_name: list() for var_name in RuntimeConfig.saved_plasticity_vars}
    for con_plastic in net.con_plastic:
        for var_name in plasticity_dict.keys():
            plasticity_dict[var_name].append(getattr(con_plastic, var_name))

    for var_name in plasticity_dict.keys():
        plasticity_dict[var_name] = np.array(plasticity_dict[var_name])

    np.savez(file_path, **plasticity_dict)


def save_instance_setup(net, performance, experiment_num=None, instance_id=None, **kwargs):
    params = flatten_dict(net.p.dict(exclude_none=True))
    experiment_type = net.p.Experiment.type
    experiment_id = net.p.Experiment.id

    folder_path_instance = get_experiment_folder(net, experiment_id, experiment_num, instance_id=instance_id)
    if not os.path.exists(folder_path_instance):
        os.makedirs(folder_path_instance)

    folder_path_experiment = get_experiment_folder(net, experiment_id, experiment_num, instance_id=None)
    file_path = join(folder_path_experiment, EXPERIMENT_SETUP_FILE_NAME[ExperimentType.INSTANCE])
    create_eval_file = not exists(file_path)

    # prepare data
    data_end = {**{'time_finished': datetime.datetime.now().strftime('%d.%m.%y - %H:%M')}, **performance}
    data_params = {name.lower().replace('.', '_'): params[name] for name in RuntimeConfig.saved_instance_params}
    data = {**{'instance_id': instance_id}, **data_params, **data_end}

    save_setup(data, experiment_num, create_eval_file=create_eval_file, do_update=False, file_path=file_path,
               save_categories=False, **kwargs)

    return experiment_num


def save_experimental_setup(net, experiment_num=None, instance_id=None, **kwargs):
    params = flatten_dict(net.p.dict(exclude_none=True))
    experiment_type = net.p.Experiment.type
    experiment_id = net.p.Experiment.id

    file_path = join(EXPERIMENT_FOLDERS[RuntimeConfig.backend], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    # add the experiment id and network type
    create_eval_file = not exists(file_path)
    last_experiment_num = get_last_experiment_num(net, experiment_id, experiment_type)
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
    folder_path = get_experiment_folder(net, experiment_id, experiment_num, instance_id=instance_id)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    data = {'experiment_id': experiment_id, 'experiment_num': f'{experiment_num:02d}',
            'network_type': str(net), 'time_finished': datetime.datetime.now().strftime('%d.%m.%y - %H:%M')}

    data = {**data, **params}

    save_setup(data, experiment_num, create_eval_file, do_update, file_path=file_path, save_categories=True, **kwargs)

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

    file_path = join(EXPERIMENT_FOLDERS[network_type.__name__], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

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
