import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

from data.nih_df_generator import NIHDfGenerator
from utils.arg_launcher import ArgLauncher

# it ignores all warnings, in this example it is useful cause MultiLabelBinarizer tells it doesn't see class
# and that is exactly what we need since we have examples with no abnormalities
warnings.filterwarnings('ignore')


class SplitterLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.description = 'Generate train and test dataframes with your custom paths to images.'
        parser.add_argument(
            '-p', '--path',
            help='Path to Nih dataset root directory.',
            required=True
        )

    def convert_args(self, args):
        args.path = Path(args.path)

    def run(self, args) -> None:
        NIHDfGenerator.generate_and_save_dfs(Path(args.path))


if __name__ == '__main__':
    SplitterLauncher(sys.argv[1:]).launch()
