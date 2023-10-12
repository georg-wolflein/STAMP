import argparse
from pathlib import Path


def _add_options(parser: argparse.ArgumentParser):
    parser.add_argument("--clini_table", type=Path, help="Path to clini_excel file", required=True)
    parser.add_argument("--slide_csv", type=Path, help="Path to slide_csv file", required=True)
    parser.add_argument("--feature_dir", type=Path, help="Path to feature directory", required=True)
    parser.add_argument("--output_path", type=Path, help="Path to output file", required=True)
    parser.add_argument("--target_label", type=str, help="Target label", required=True)
    parser.add_argument("--cat_labels", type=str, nargs="+", default=[], help="Category labels")
    parser.add_argument("--cont_labels", type=str, nargs="+", default=[], help="Continuous labels")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Categories")

def add_commands(subparsers):
    # add train command
    train_parser = subparsers.add_parser("train", description="Run full training instead of cross-validation")
    _add_options(train_parser)
    def train_callback(args):
        from .marugoto.transformer.helpers import train_categorical_model_
        train_categorical_model_(clini_table=args.clini_table, 
                                slide_csv=args.slide_csv,
                                feature_dir=args.feature_dir, 
                                output_path=args.output_path,
                                target_label=args.target_label, 
                                cat_labels=args.cat_labels,
                                cont_labels=args.cont_labels, 
                                categories=args.categories)

    # add crossval command
    crossval_parser = subparsers.add_parser("crossval", description="Run cross validation for n_splits models")
    _add_options(crossval_parser)
    crossval_parser.add_argument("--n_splits", type=int, default=5, help="Number of splits")
    def crossval_callback(args):
        from .marugoto.transformer.helpers import categorical_crossval_
        categorical_crossval_(clini_table=args.clini_table, 
                            slide_csv=args.slide_csv,
                            feature_dir=args.feature_dir,
                            output_path=args.output_path,
                            target_label=args.target_label,
                            cat_labels=args.cat_labels,
                            cont_labels=args.cont_labels,
                            categories=args.categories,
                            n_splits=args.n_splits)

    # add deploy command
    deploy_parser = subparsers.add_parser("deploy", description="Deploy model on data")
    _add_options(deploy_parser)
    deploy_parser.add_argument("--model_path", type=Path, help="Path to model .pkl to deploy")
    def deploy_callback(args):
        from .marugoto.transformer.helpers import deploy_categorical_model_
        deploy_categorical_model_(clini_table=args.clini_table,
                                slide_csv=args.slide_csv,
                                feature_dir=args.feature_dir,
                                model_path=args.model_path,
                                output_path=args.output_path,
                                target_label=args.target_label,
                                cat_labels=args.cat_labels,
                                cont_labels=args.cont_labels)
    
    return {
        "train": train_callback,
        "crossval": crossval_callback,
        "deploy": deploy_callback
    }


def main():
    parser = argparse.ArgumentParser(
        description='Associative modeling with a Vision Transformer.')
    
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    subcommand_callbacks = add_commands(subparsers)
    args = parser.parse_args()
    subcommand_callbacks[args.subcommand](args)


if __name__ == "__main__":
    main()