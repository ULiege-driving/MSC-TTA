import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs='+', help="Path to the dataset folder", required=True)
parser.add_argument("--teacher", help="teacher network name", default='segformer', choices=['setr', 'segformer', 'pspnet', 'pointrend_seg'])
parser.add_argument("--batchsize", help="batch_size", type=int, default=1)

args = parser.parse_args()
