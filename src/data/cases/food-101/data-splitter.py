import argparse
import os
import random
import json
from shutil import copyfile


def create_argument_parser():
    parser = argparse.ArgumentParser(description='parse incoming')

    parser.add_argument('-p', '--path',
                        help="The path to the input files",
                        required=True,
                        default='.',
                        type=str)

    parser.add_argument('-o', '--output',
                        help="The path to save the new input files",
                        required=True,
                        type=str)

    parser.add_argument('-i', '--train-percent',
                        default=1.0,
                        help="The percentage of training images",
                        type=float)

    parser.add_argument('-v', '--validation-split',
                        default=0.1,
                        help="The percentage of training used as validation set",
                        type=float)

    parser.add_argument('-t', '--test-percent',
                        default=1.0,
                        help="The percentage of test images",
                        type=float)
    return parser


def load_data(filename, labels=None, split_ratio=0.1):
    with open(filename, 'r') as file_content:
        content = json.load(file_content)
        return split_set(content, split_ratio, labels)


def split_set(content, ratio, labels=None):
    temp_data = dict(content)
    s1 = {}
    s2 = {}
    for label, data in temp_data.items():
        if labels and label not in labels:
            continue

        random.shuffle(data)
        split_point = int(len(data) * ratio)
        s1[label] = data[:split_point]
        s2[label] = data[split_point:]

    return s1, s2
    

def export(src_path, out_path, train=None, validation=None, test=None,
        ext='jpg'):
    metadata_path, data_path = create_directory_structure(out_path)
    metadata = {'train': export_set(f"{src_path}/images",
                                    f"{data_path}/train",
                                    train,
                                    ext),
                'test': export_set(f"{src_path}/images",
                                   f"{data_path}/test",
                                   test,
                                   ext),
                'validation': export_set(f"{src_path}/images",
                                         f"{data_path}/validation",
                                         validation,
                                         ext)
                }

    for k, v in metadata.items():
        metadata_file = f"{metadata_path}/{k}.txt"
        with open(metadata_file, 'w') as outfile:
            outfile.write('\n'.join(v))


def export_set(src_path, out_path, data, ext='jpg'):
    metadata = []
    for label, entry in data.items():
        dst_dir = os.path.join(out_path, label)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for item in entry:
            _, filename = parse_filename(item)
            src_file = f"{os.path.join(src_path, label, filename)}.{ext}"
            dst_file = f"{os.path.join(out_path, label, filename)}.{ext}"
            copyfile(src_file, dst_file)
            metadata.append(f"{label}/{filename}")
    return metadata


def parse_filename(entry):
    s = entry.split('/')
    label = s[0]
    filename = s[1]
    return label, filename


def create_directory_structure(out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    metadata_path = os.path.join(out_path, "meta")

    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)

    data_path = os.path.join(out_path, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    return metadata_path, data_path

binary_dataset = ['caesar_salad', 'chicken_wings']

four_classes =  ['apple_pie', 'beef_tartare', 'caesar_salad', 'chicken_wings']

nine_classes = ['apple_pie', 'beef_tartare', 'pizza', 
                'bruschetta', 'caesar_salad', 'carrot_cake',
                'cheesecake', 'chicken_curry', 'chicken_wings']

twelve_classes = ['apple_pie', 'beef_carpaccio','beef_tartare',
                  'beet_salad', 'pizza', 'bruschetta',
                  'caesar_salad', 'carrot_cake', 'cheese_plate', 
                  'cheesecake', 'chicken_curry', 'chicken_wings']

if __name__ == '__main__':
    cmdline_args = create_argument_parser().parse_args()

    input_path = cmdline_args.path
    output_path = cmdline_args.output
    
    ## No filter on classes
    ## implies on using the whole dataset
    # target_labels = None 
    target_labels = four_classes

    train_sample, _ = load_data(f"{cmdline_args.path}/meta/train.json",
                                split_ratio=cmdline_args.train_percent,
                                labels=target_labels)

    valid_sample, train_sample = split_set(train_sample,
                                           cmdline_args.validation_split)

    test_sample, _ = load_data(f"{cmdline_args.path}/meta/test.json",
                               split_ratio=cmdline_args.test_percent,
                               labels=target_labels)
    export(cmdline_args.path,
           cmdline_args.output,
           train_sample,
           test_sample,
           valid_sample
    )

    print(f"Data set exported to {cmdline_args.output}")

