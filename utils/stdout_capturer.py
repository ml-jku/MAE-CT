from io import StringIO
import sys
import logging

# modification of https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class StdoutCapturer:
    """ captures stdout and passes it to logging.info instead """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        for line in self._stringio.getvalue().splitlines():
            logging.info(line)
        del self._stringio
        sys.stdout = self._stdout