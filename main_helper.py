import sys
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

cli = None
class CLI:
    def __init__(self, *, invokedAs=None):
        self.invokedAs = invokedAs
        self.processName = None
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
        self.loggingFormat = ''.join([
            ('' if self.processName is None else self.processName.replace('%', '%%') + ': '),
            '%(asctime)s %(levelname)s:%(name)s:%(message)s',
        ])

    def setupLogging(self):
        logging.basicConfig(format=self.loggingFormat)

def setupCLI():
    global cli
    cli = CLI(invokedAs=sys.argv[0])
    cli.setupLogging()
    return cli

def no_main():
    raise RuntimeError("This module can't be used as script, main or entry point.")

if __name__ == '__main__':
    no_main()
