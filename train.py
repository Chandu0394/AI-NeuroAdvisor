import os
import shutil
import argparse
import logging
from tqdm import tqdm
from glob import glob

from utils.parser import ParserUse

from get_resnet import features
from train_resnet_con import train_resnet
from train_fusion import train_trans



def run_all(args):
    args = train_resnet(args)
    args = train_resnet(args)
    args = features(args)
    args = train_trans(args)
    logging.info("\n\n" + "=="*10)
    logging.info(args)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", default="train", type=str, required=True)
    args.add_argument("-n", default="", type=str, help="Notes for training")
    args = args.parse_args()
    args = ParserUse(args.cfg, "train_all").add_args(args)

    logging.info("Parameters saved to " + ">>" * 10 + args.log_time)
    run_all(args)
