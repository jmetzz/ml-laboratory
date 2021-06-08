import argparse
from pathlib import Path
from shutil import copyfile

from utils.data_helper import split_set
from utils.data_loader import load_json_file


def create_argument_parser():
    parser = argparse.ArgumentParser(description="parse incoming")

    parser.add_argument(
        "-p",
        "--path",
        help="The path to the input files",
        required=True,
        default=".",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to save the new input files",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-i",
        "--train-percent",
        default=1.0,
        help="The percentage of training images",
        type=float,
    )

    parser.add_argument(
        "-v",
        "--validation-split",
        default=0.1,
        help="The percentage of training used as validation set",
        type=float,
    )

    parser.add_argument(
        "-t",
        "--test-percent",
        default=1.0,
        help="The percentage of test images",
        type=float,
    )
    return parser


def export(src_path: str, out_path: str, train=None, validation=None, test=None, ext="jpg"):
    metadata_path, data_path = create_directory_structure(Path(out_path))
    metadata = {
        "train": export_set(f"{src_path}/images", f"{data_path}/train", train, ext),
        "test": export_set(f"{src_path}/images", f"{data_path}/test", test, ext),
        "validation": export_set(f"{src_path}/images", f"{data_path}/validation", validation, ext),
    }

    for key, value in metadata.items():
        with Path(metadata_path, f"{key}.txt").open(mode="w") as outfile:
            outfile.write("\n".join(value))


def export_set(src_path, out_path, data, ext="jpg"):
    metadata = []
    for label, entry in data.items():
        dst_dir = Path(out_path, label)
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)

        for item in entry:
            _, filename = parse_filename(item)
            src_file = Path(src_path, label, f"{filename}.{ext}")
            dst_file = Path(out_path, label, f"{filename}.{ext}")
            copyfile(src_file, dst_file)
            metadata.append(f"{label}/{filename}")
    return metadata


def parse_filename(entry):
    parts = entry.split("/")
    label = parts[0]
    filename = parts[1]
    return label, filename


def create_directory_structure(out_path: Path) -> tuple[Path, Path]:
    metadata_path = Path(out_path, "meta")
    if not metadata_path.exists():
        metadata_path.mkdir(parents=True)

    data_path = Path(out_path, "data")
    if not data_path.exists():
        data_path.mkdir(parents=True)

    return metadata_path, data_path


binary_dataset = ["caesar_salad", "chicken_wings"]

four_classes = ["apple_pie", "beef_tartare", "caesar_salad", "chicken_wings"]

nine_classes = [
    "apple_pie",
    "beef_tartare",
    "pizza",
    "bruschetta",
    "caesar_salad",
    "carrot_cake",
    "cheesecake",
    "chicken_curry",
    "chicken_wings",
]

twelve_classes = [
    "apple_pie",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "pizza",
    "bruschetta",
    "caesar_salad",
    "carrot_cake",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_wings",
]

if __name__ == "__main__":
    cmdline_args = create_argument_parser().parse_args()

    input_path = cmdline_args.path
    output_path = cmdline_args.output

    # No filter on classes
    # implies on using the whole dataset
    # target_labels = None
    target_labels = four_classes

    train_sample, _ = load_json_file(
        f"{cmdline_args.path}/meta/train.json",
        split_ratio=cmdline_args.train_percent,
        labels=target_labels,
    )

    valid_sample, train_sample = split_set(train_sample, cmdline_args.validation_split)

    test_sample, _ = load_json_file(
        f"{cmdline_args.path}/meta/test.json",
        split_ratio=cmdline_args.test_percent,
        labels=target_labels,
    )
    export(cmdline_args.path, cmdline_args.output, train_sample, test_sample, valid_sample)

    print(f"Data set exported to {cmdline_args.output}")
