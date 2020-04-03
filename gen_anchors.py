from kmeans import AnchorKmeans
from datasets import AnnotParser
import argparse


def main(args):
    file_type = args["type"]
    k = args["k_clusters"]
    annot_dir = args["dir_path"]
    parser = AnnotParser(file_type)

    print("[INFO] Load datas from {}".format(annot_dir))
    boxes = parser.parse(annot_dir)

    print("[INFO] Initialize model")
    model = AnchorKmeans(k)

    print("[INFO] Training...")
    model.fit(boxes)

    anchors = model.anchors_
    print("[INFO] The results anchors:\n{}".format(anchors))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d",
                    "--dir_path",
                    required=True,
                    help="directory path of annotation files")
    ap.add_argument("-t",
                    "--type",
                    choices=['xml', 'json', 'csv'],
                    default='xml',
                    help="type of annotation file")
    ap.add_argument("-k",
                    "--k_clusters",
                    type=int,
                    default=5,
                    help="the number of clusters")
    args = vars(ap.parse_args())
    main(args)
