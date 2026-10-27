"""python main.py --invoice_path=data/invoices/invoice_1001.txt"""

import sys

from invoice_agents.cli import main

if __name__ == "__main__":
    sys.exit(main())
