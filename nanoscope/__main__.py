"""`python -m nanoscope` (M5-T01).

The same entry point the `nanoscope` console script uses, so a checkout without
an installed script still starts the application the same way.
"""

import sys

from nanoscope.app.main import main

if __name__ == "__main__":
    sys.exit(main())
