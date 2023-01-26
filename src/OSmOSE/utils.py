import shutil
import os

def display_folder_storage_infos(dir_path: str) -> None:

    usage=shutil.disk_usage(dir_path)
    print("Total storage space (TB):",round(usage.total / (1024**4),1))
    print("Used storage space (TB):",round(usage.used / (1024**4),1))
    print('-----------------------')
    print("Available storage space (TB):",round(usage.free / (1024**4),1))
    
def list_not_built_datasets(datasets_folder_path: str) -> None:
    """Prints the available datasets that have not been built by the `Dataset.build()` function.
    
        Parameter:
        ----------
        dataset_folder_path: The path to the directory containing the datasets"""

    dataset_list = [directory for directory in sorted(os.listdir(datasets_folder_path)) if os.path.isdir(directory) ]

    list_not_built_datasets = []

    for dataset_directory in dataset_list:

        if os.path.exists(os.path.join(datasets_folder_path,dataset_directory,'raw/audio/original/') ):
            list_not_built_datasets.append(dataset_directory)

    print("List of the datasets not built yet:")

    for dataset in list_not_built_datasets:
        print("  - {}".format(dataset))    
        