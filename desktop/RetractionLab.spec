# Build with the Python interpreter for the target OS. Linux cannot cross-freeze
# a Windows executable; the Windows workflow runs this same specification.
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent
sys.path.insert(0, str(root / 'backend'))
hiddenimports = collect_submodules('retraction') + [
    'uvicorn.logging', 'uvicorn.loops.asyncio', 'uvicorn.protocols.http.h11_impl',
    'uvicorn.lifespan.on', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
]
datas = [
    (str(root / 'frontend' / 'dist'), 'frontend/dist'),
    (str(root / 'desktop' / 'assets'), 'desktop/assets'),
    (str(root / 'desktop' / 'licenses'), 'desktop/licenses'),
    (str(root / 'docs'), 'docs'),
    (str(root / 'README.md'), '.'),
    (str(root / 'build-info.json'), '.'),
    # The environment records a hash of these exact source bytes. Keep them
    # beside the frozen module so packaged replay provenance matches source.
    (str(root / 'backend' / 'retraction' / 'environment.py'), 'retraction'),
    (str(root / 'backend' / 'retraction' / 'tissue.py'), 'retraction'),
]
datas.extend((str(path), '.') for path in (root / 'desktop' / 'runtime-ca').glob('*.pem'))
a = Analysis([str(root / 'desktop' / 'app.py')],
             pathex=[str(root), str(root / 'backend')],
             binaries=[], datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['tkinter', 'matplotlib', 'PyQt5', 'PyQt6', 'pytest', 'IPython', 'notebook'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='RetractionLab',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False,
          icon=str(root / 'desktop' / 'assets' / 'icon.ico') if sys.platform == 'win32' else None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='RetractionLab')
