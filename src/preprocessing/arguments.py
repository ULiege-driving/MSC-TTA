import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs='+', help="Path to the dataset folder", required=True)
parser.add_argument("--teacher", help="teacher network name", default='segformer', choices=['setr', 'segformer', 'pspnet', 'pointrend_seg'])
parser.add_argument("--batchsize", help="batch_size", type=int, default=1)
parser.add_argument("--start", help="Only select a subset of the dataset (starting point)", type=int, default=None)
parser.add_argument("--stop", help="Only select a subset of the dataset (end point)", type=int, default=None)

args = parser.parse_args()
