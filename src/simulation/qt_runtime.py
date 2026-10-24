"""Reuse V2's local Cocoa plugin workaround without changing system settings."""
import atexit
import os
from pathlib import Path
import shutil
import sys
import tempfile


def prepare_qt():
    if sys.platform!='darwin' or os.getenv('QT_QPA_PLATFORM_PLUGIN_PATH'): return
    import PySide6
    source=Path(PySide6.__file__).parent/'Qt/plugins/platforms/libqcocoa.dylib'
    if source.exists():
        directory=Path(tempfile.mkdtemp(prefix='robot-visual-qt-'))
        shutil.copyfile(source,directory/source.name)
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(directory)
        atexit.register(shutil.rmtree,directory,ignore_errors=True)
