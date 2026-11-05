# Qt and Chromium notices

Retraction Lab's desktop distribution uses PySide6, Shiboken6 and Qt 6.8.3, including Qt WebEngine and Chromium. The Qt libraries are distributed as separate dynamic libraries in the application folder. Their upstream copyright and license notices apply.

The text files in this directory are copied verbatim from the official Qt WebEngine `v6.8.3` source tag. `sources.json` records the source URL and SHA256 of each file. PySide6's official wheel metadata declares `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`; the presence of several upstream license texts here preserves those notices and does not change the license of any component.

The LGPLv3 text incorporates GPLv3; both texts are included. `LICENSE.Chromium` contains the root Chromium BSD notice. Chromium also contains third-party components under their respective licenses; that root notice alone is not a complete list of their acknowledgments. Qt's official component-specific attribution documentation is at <https://doc.qt.io/qt-6.8/qtwebengine-licensing.html>.

Exact upstream source tags:

- PySide6 and Shiboken6: <https://github.com/qt/pyside-setup/tree/v6.8.3>
- Qt Base: <https://github.com/qt/qtbase/tree/v6.8.3>
- Qt WebEngine: <https://github.com/qt/qtwebengine/tree/v6.8.3>
- Qt WebEngine's Chromium dependency is pinned by the `src/3rdparty` gitlink in that Qt WebEngine tag.

The source for the application and its repeatable build recipe is in this repository. Rebuilding the application with a modified compatible Qt/PySide6 library is supported by the source build workflow; keep the complete dynamic-library folder with the executable.
