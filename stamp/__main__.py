import argparse
from typing import Mapping, Callable

from .preprocessing.wsi_norm import add_commands as add_preprocessing_commands
from .modeling.modeling import add_commands as add_modeling_commands
from .modeling.statistics import add_commands as add_statistics_commands

parser = argparse.ArgumentParser(prog="stamp", description="STAMP: Solid Tumor Associative Modeling in Pathology")
subcommands = parser.add_subparsers(title="subcommands", dest="subcommand")

# Add the subcommands. The callbacks dict maps each subcommand to a function that takes the parsed arguments and performs the corresponding action.
callbacks: Mapping[str, Callable[[argparse.Namespace], None]] = {
    **add_preprocessing_commands(subcommands), # preprocess
    **add_modeling_commands(subcommands), # train, crossval, deploy
    **add_statistics_commands(subcommands) # roc
}

args = parser.parse_args()
subcommand = args.subcommand

if subcommand is None:
    parser.print_help()
elif subcommand in callbacks:
    callbacks[subcommand](args)
else:
    raise ValueError(f"Unknown subcommand {subcommand}")