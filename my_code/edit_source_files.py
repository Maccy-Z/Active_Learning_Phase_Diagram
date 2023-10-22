assert False, "Remove this import, legacy file"
import sys
import os
import shutil

print(sys.executable)
print(sys.version)

src = "/opt/project/SheffieldML-GPy-e91799a/GPy/core/gp.py"
dst = "/usr/local/lib/python3.11/site-packages/GPy-1.12.0-py3.11-linux-x86_64.egg/GPy/core/gp.py"
shutil.copy2(src, dst)

exit(5)