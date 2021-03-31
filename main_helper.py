import sys
import argparse
from pathlib import Path
import logging

def getLoggerName(*, name, package, file):
    if name != '__main__':
        return name
    basename = Path(file).stem
    if package is not None:
        if package == '':
            return basename
        return '.'.join([package, basename])
    invokedAs = sys.argv[0]
    if invokedAs is not None and invokedAs not in ["", "-c"]:
        if invokedAs == file:
            return basename
    return name

loggerName = getLoggerName(name=__name__, package=__package__, file=__file__)
logger = logging.getLogger(loggerName)


cli = None
class CLI:
    def __init__(self, *, invokedAs=None):
        self.invokedAs = invokedAs
        self.processName = None
        self.argvParser = None
        if self.invokedAs is None:  # Unknown.
            # No prefix, for now.
            pass
        elif self.invokedAs == "":  # Interactive session (python prompt).
            self.processName = "(python interactive)"
        elif self.invokedAs == "-c":  # Command-line code.
            self.processName = "(python code argument)"
        else:  # Assume path to script. With -m, it's an absolute path, otherwise whatever the user typed/completed.
            # Use script's basename (without .py extension) as prefix.
            self.processName = Path(self.invokedAs).stem
        self.loggingLevel = None
        self.loggingLevelNames = ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.loggingPrefix = '' if self.processName is None else self.processName + ': '
        self.loggingHaveTime = None
        self.loggingFormat = None

    def setupLogging(self):
        self.loggingFormat = (
            self.loggingPrefix.replace('%', '%%') +
            ('%(asctime)s ' if self.loggingHaveTime else '') +
            '%(levelname)s:%(name)s:%(message)s'
        )
        logging.basicConfig(format=self.loggingFormat, level=self.loggingLevel)

    @classmethod
    def createArgvParser(cls, **kwargs):
        """Create an argparse.ArgumentParser with certain defaults.
        Additionally, all keyword arguments given here are forwarded to the ctor.

        Other parts of the code should use this (main_helper.CLI.createArgvParser())
        to get a parser to pass into main_helper.setupCLI().
        """
        forward_kwargs = {
            # We need this to avoid "ps-auxw" being line-broken at the dash.
            # Newlines, any amount of them, seem to be replaced by a single space
            # if we would leave the default, but we want to force a line break
            # to avoid the unfortunate one.
            'formatter_class': argparse.RawDescriptionHelpFormatter,
            'epilog': "CLI-P is commandline driven semantic image search using OpenAI's CLIP."
                "\nBy ps-auxw. OO/GUI work by canvon.",
        }
        forward_kwargs.update(kwargs)
        return argparse.ArgumentParser(**forward_kwargs)

    def setupArgvParser(self, parser=None):
        if self.argvParser is not None:
            raise RuntimeError("argv parser already set up")
        if parser is None:
            parser = self.createArgvParser()
        self.argvParser = parser

        self.argvParser.add_argument('--log-level', dest='loggingLevel', default=None,
            choices=self.loggingLevelNames)

        self.argvParser.add_argument('--log-timestamp', '--log-ts', dest='log_ts', action='store_const', const=True,
            help="Have a timestamp with each logged message.")
        self.argvParser.add_argument('--no-log-timestamp', '--no-log-ts', dest='log_ts', action='store_const', const=False)

    def runArgvParser(self):
        if self.argvParser is None:
            raise RuntimeError("argv parser missing, please call setupArgvParser(), first")
        self.args = self.argvParser.parse_args()

        if 'loggingLevel' in self.args and self.args.loggingLevel is not None:
            level = self.args.loggingLevel
            if level not in self.loggingLevelNames:
                print(self.loggingPrefix + f'error: Invalid logging level {level!r} requested', file=sys.stderr, flush=True)
                sys.exit(2)
            self.loggingLevel = getattr(logging, level)
        if 'log_ts' in self.args and self.args.log_ts is not None:
            self.loggingHaveTime = self.args.log_ts

def setupCLI(*, argvParser=None):
    global cli
    cli = CLI(invokedAs=sys.argv[0])
    cli.setupArgvParser(argvParser)
    cli.runArgvParser()
    cli.setupLogging()
    return cli

def no_main():
    raise RuntimeError("This module can't be used as script, main or entry point.")

def example_main():
    setupCLI()
    print("Doing some example logging..."
        " (debug, info, warning, error, critical; of which"
        " everything below warning will likely not be passed through.)", flush=True)

    x = 3
    y = 7
    logger.debug("This is an example debug-level message. %d + %d == %d", x, y, x + y)

    logger.info("This is an example info-level message: Going up through the logging levels!")
    logger.warning("Attention! Here is an example warning-level message.")
    logger.error("The following example error has occurred: ...")
    logger.critical("Critical example situation...")

    print("End of example logging.", flush=True)

if __name__ == '__main__':
    example_main()
