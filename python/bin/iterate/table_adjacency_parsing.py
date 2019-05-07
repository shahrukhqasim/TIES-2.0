import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__),'../../')))

import argparse
from iterators.table_adjacency_parsing_iterator import TableAdjacencyParsingIterator
from libs.configuration_manager import ConfigurationManager as gconfig

if __name__ != "__main__":
    print("Execute as a python script. This is not an importable module.")
    exit(0)


parser = argparse.ArgumentParser(description='Run training/testing/validation for graph based clustering')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--test', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--profile', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--visualize', default=False, help="Whether to run layer wise visualization (x-mode only)")
args = parser.parse_args()


gconfig.init(args.input, args.config)

trainer = TableAdjacencyParsingIterator()

if args.test:
    trainer.test()
elif args.profile:
    trainer.profile()
elif args.visualize:
    trainer.visualize()
else:
    trainer.train()
