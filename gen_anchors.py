from kmeans import AnchorKmeans
from datasets import parse_xml
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to annotations directory")
args = vars(ap.parse_args())
