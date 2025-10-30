from osekit.public_api.export_analysis import create_parser


def test_parser_factory() -> None:
    parser = create_parser()

    assert parser.description

    expected_args = {
        "--analysis",
        "--ads-json",
        "--sds-json",
        "--subtype",
        "--matrix-folder-path",
        "--spectrogram-folder-path",
        "--welch-folder-path",
        "--first",
        "--last",
        "--downsampling-quality",
        "--upsampling-quality",
        "--umask",
        "--tqdm-disable",
        "--multiprocessing",
        "--use-logging-setup",
        "--nb-processes",
        "--dataset-json-path",
    }

    actual_args = {
        action.option_strings[0] for action in parser._actions if action.option_strings
    }
    assert expected_args.issubset(actual_args)


def test_argument_defaults() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "--analysis",
            "1",
        ],
    )

    assert args.analysis == 1
    assert args.ads_json is None
    assert args.sds_json is None
    assert args.subtype is None
    assert args.matrix_folder_path is None
    assert args.spectrogram_folder_path is None
    assert args.welch_folder_path is None
    assert args.first == 0
    assert args.last == -1
    assert args.downsampling_quality is None
    assert args.upsampling_quality is None
    assert args.umask == 0o002  # noqa: PLR2004
    assert args.tqdm_disable == "true"
    assert args.multiprocessing == "false"
    assert args.use_logging_setup == "false"
    assert args.nb_processes is None
    assert args.dataset_json_path is None


def test_specified_arguments() -> None:
    parser = create_parser()

    arguments = {
        "--analysis": 2,
        "--ads-json": r"path/to/ads.json",
        "--sds-json": r"path/to/ads.json",
        "--subtype": "FLOAT",
        "--matrix-folder-path": r"out/matrix",
        "--spectrogram-folder-path": r"out/spectro",
        "--welch-folder-path": r"out/welch",
        "--first": 10,
        "--last": 12,
        "--downsampling-quality": "HQ",
        "--upsampling-quality": "VHQ",
        "--umask": 0o022,
        "--tqdm-disable": "False",
        "--multiprocessing": "True",
        "--nb-processes": 3,
        "--use-logging-setup": "True",
        "--dataset-json-path": r"path/to/dataset.json",
    }

    args = parser.parse_args(
        [str(arg_part) for arg in arguments.items() for arg_part in arg],
    )

    assert args.analysis == arguments["--analysis"]
    assert args.ads_json == arguments["--ads-json"]
    assert args.sds_json == arguments["--sds-json"]
    assert args.subtype == arguments["--subtype"]
    assert args.matrix_folder_path == arguments["--matrix-folder-path"]
    assert args.spectrogram_folder_path == arguments["--spectrogram-folder-path"]
    assert args.welch_folder_path == arguments["--welch-folder-path"]
    assert args.first == arguments["--first"]
    assert args.last == arguments["--last"]
    assert args.downsampling_quality == arguments["--downsampling-quality"]
    assert args.upsampling_quality == arguments["--upsampling-quality"]
    assert args.umask == arguments["--umask"]
    assert args.tqdm_disable == arguments["--tqdm-disable"]
    assert args.multiprocessing == arguments["--multiprocessing"]
    assert args.use_logging_setup == arguments["--use-logging-setup"]
    assert args.nb_processes == arguments["--nb-processes"]
    assert args.dataset_json_path == arguments["--dataset-json-path"]
