from abc import abstractmethod, ABCMeta
from argparse import ArgumentParser


class ArgLauncher(metaclass=ABCMeta):

    def __init__(self, args, parser=ArgumentParser()):
        self._args = args
        self._parser = parser

    @property
    def parser(self) -> ArgumentParser:
        return self._parser

    @property
    def args(self):
        return self._args

    def launch(self) -> None:
        """
        Runs launcher.
        @return: None
        """
        self.setup_parser(self._parser)
        args = self._parser.parse_args(self._args)
        self.validate_args(args)
        self.convert_args(args)
        self.run(args)

    @abstractmethod
    def setup_parser(self, parser: ArgumentParser) -> None:
        """
        Base method for argparse parser setup.
        @param parser: Parser to be initialized.
        @return: None
        """
        raise NotImplementedError()

    def validate_args(self, args):
        """
        Base method which allows subclasses to validate supplied input from CLI.
        @param args: Object with CLI args.
        @return: None
        """
        pass

    def convert_args(self, args):
        """
        Allows user to convert supplied args into different form.
        @param args: Object with CLI args.
        @return: None
        """
        pass

    @abstractmethod
    def run(self, args) -> None:
        """
        Execute arbitrary code with supplied CLI args.
        @param args: Object with CLI args.
        @return: None
        """
        raise NotImplementedError()
