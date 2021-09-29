import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

from data.nih_dataset import NIHDataset
from utils.arg_launcher import ArgLauncher

# it ignores all warnings, in this example it is useful cause MultiLabelBinarizer tells it doesn't see class
# and that is exactly what we need since we have examples with no abnormalities
warnings.filterwarnings('ignore')


class SplitterLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.description = 'Generate train and test dataframes with your custom paths to images.'
        parser.add_argument('-p', '--path',
                            help='Path to Nih dataset root directory.',
                            required=True)
        parser.add_argument('-o', '--out-dir',
                            help='Output dir for generated data.',
                            default='df_split_files', type=str)
        parser.add_argument('--prefix',
                            help='Name prefix for generated files.',
                            default='official', type=str)
        parser.add_argument('--val-size',
                            help='Validation split size in %.',
                            default=0.0, type=float)
        parser.add_argument('-c', '--classes',
                            help='Path to file with classes in yaml format. '
                                 'Allows to select which classes will be included output files.',
                            default=None, type=str)

    def convert_args(self, args):
        args.path = Path(args.path)

    def run(self, args) -> None:
        classes_file = args.classes

        NIHDataset.save_dfs(
            dataset_path=Path(args.path),
            out_dir=Path(args.path) / args.out_dir,
            name_prefix=args.prefix,
            validation_df_size=args.val_size,
            classes=OmegaConf.load(classes_file) if classes_file else None
        )


if __name__ == '__main__':
    SplitterLauncher(sys.argv[1:]).launch()
