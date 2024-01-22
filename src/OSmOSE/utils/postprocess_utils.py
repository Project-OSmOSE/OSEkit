import os
import shutil
import glob


def list_datasets(path_osmose_dataset):
    print("Available datasets:")
    dataset_list = sorted(os.listdir(path_osmose_dataset))
    for dataset in dataset_list:
        print(f"  - {dataset}")


def check_available_file_resolution(path_osmose_dataset, campaign_ID, dataset_ID):

    base_path = os.path.join(path_osmose_dataset, campaign_ID, dataset_ID, 'data', 'audio')
    dirname = os.listdir(base_path)

    print(f'Dataset : {campaign_ID}/{dataset_ID}')
    print('Available Resolution (LengthFile_samplerate) :', end='\n')

    [print(f' {d}') for d in dirname]
    return dirname


def extract_config(path_osmose_dataset, list_campaign_ID, list_dataset_ID, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for campaign_ID, dataset_ID in zip(list_campaign_ID, list_dataset_ID):

        dataset_resolution = check_available_file_resolution(path_osmose_dataset, campaign_ID, dataset_ID)

        for dr in dataset_resolution:
            #  audio config files
            path1 = os.path.join(path_osmose_dataset, campaign_ID, dataset_ID, 'data', 'audio', dr)
            files1 = glob.glob(os.path.join(path1, '**.csv'))

            full_path1 = os.path.join(out_dir, 'export_' + dataset_ID, dr)
            if not os.path.exists(full_path1):
                os.makedirs(full_path1)
            [shutil.copy(file, full_path1) for file in files1]

        #  spectro config files
        path2 = os.path.join(path_osmose_dataset, campaign_ID, dataset_ID, 'processed', 'spectrogram')
        files2 = []
        for root, dirs, files in os.walk(path2):
            files2.extend([os.path.join(root, file) for file in files if file.lower().endswith('.csv')])

        full_path2 = os.path.join(out_dir, 'export_' + dataset_ID, 'spectro')
        if not os.path.exists(full_path2):
            os.makedirs(full_path2)
        [shutil.copy(file, full_path2) for file in files2]

    print(f'\nFiles exported to {out_dir}')
