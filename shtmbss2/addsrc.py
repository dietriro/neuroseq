import sys
import importlib.util

spec_static = importlib.util.find_spec("_static")
if spec_static is None:
    sys.path.append('/opt/app-root/src/drive/My Libraries/My Library/shtm4pynn/src')
