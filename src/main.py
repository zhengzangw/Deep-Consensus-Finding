# /usr/bin/env python3

import argparse


def parseargs(arg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="default", type=str, help="Name of experiments.")
    args = parser.parse_args(arg)
    return args


if __name__ == "__main__":
    args = parseargs()
