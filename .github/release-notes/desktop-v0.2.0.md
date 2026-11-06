Retraction Lab is a self-contained Windows x64 research application with an
academic procedure interface, persistent simulated tissue injury, bleeding and
bile leakage, reinforcement-learning feedback, recorded episodes and exports.

Download **RetractionLab-0.2.0-windows-x64-Setup.exe** below and run it to install
for your user account. Python, Node and a separate browser are not required.
The accompanying `.sha256` file and `build-manifest.json` identify the installer.

The application payload was built from source commit
`e62f26c44d964a360e057338ad8ce7c6a4e039b1` and passed
[native Windows interface, model, installation and uninstallation checks](https://github.com/returnedlegend/Medical-AI/actions/runs/42085408337).
This release preserves those exact verified installer bytes. Its download is a
standard Release asset; GitHub Actions artifact sign-in and retention rules do
not apply to this asset.

- [Installation and first procedure](https://github.com/returnedlegend/Medical-AI/blob/e62f26c44d964a360e057338ad8ce7c6a4e039b1/docs/desktop.md)
- [Tool/body response and reward mechanisms](https://github.com/returnedlegend/Medical-AI/blob/e62f26c44d964a360e057338ad8ce7c6a4e039b1/docs/tissue-response.md)
- [Fixed-policy evaluation](https://github.com/returnedlegend/Medical-AI/blob/e62f26c44d964a360e057338ad8ce7c6a4e039b1/docs/rl-final-001.md)

The installer is unsigned. Tissue mechanics and physiology are illustrative,
unvalidated research models, with no claim of clinical safety or surgical
competence. The optional Linux SOFA runtime is not included in this Windows
package. Studies are stored separately from the installation directory and are
preserved during uninstallation.
