from kmeans import AnchorKmeans
from datasets import parse_xml
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())