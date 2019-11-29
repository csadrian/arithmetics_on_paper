import os
import argparse

def str2bool(v):
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('--dataset', dest="dataset", default="test1", type=str, help="Dataset, loads tfrecords postfixed with '.(train|val|test).tfrecords'.")
parser.add_argument('--dataset_path', dest="dataset_path", default="datasets", type=str, help="Path for tfrecords.")
parser.add_argument('--split', dest="split", default="test", type=str, help="Split to use at evaluation")
parser.add_argument('--eval_size', dest="eval_size", default=1000, type=str, help="Number of problems used for evaluation")

# model paths
parser.add_argument('--prefix', dest="prefix", default="trash", help="File prefix for the output visualizations and models.")
parser.add_argument('--model_path', dest="model_path", default=None, help="Path to saved networks. If None, build networks from scratch.")

args = parser.parse_args()

def getArgs():
    # put output files in a separate directory
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    return args