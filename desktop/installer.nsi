; NSIS 3.11. Per-user application files; studies live in a separate user-data
; directory and are deliberately absent from the uninstallation manifest.
Unicode True
!include "MUI2.nsh"
!include "x64.nsh"
!include "WinVer.nsh"

Name "Retraction Lab"
OutFile "${INSTALLER_OUT}"
InstallDir "$LOCALAPPDATA\Programs\Retraction Lab"
InstallDirRegKey HKCU "Software\RetractionLab\Installer" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 32
VIProductVersion "${VERSION}.0"
VIAddVersionKey /LANG=1033 "ProductName" "Retraction Lab"
VIAddVersionKey /LANG=1033 "FileDescription" "Retraction Lab desktop installer"
VIAddVersionKey /LANG=1033 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1033 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=1033 "LegalCopyright" "Retraction Lab contributors"
BrandingText "Retraction Lab ${VERSION}"

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\RetractionLab.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open Retraction Lab"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "Retraction Lab requires 64-bit Windows 10 or later."
    Abort
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_OK|MB_ICONSTOP "Retraction Lab requires Windows 10 or later."
    Abort
  ${EndIf}
FunctionEnd

Section "Retraction Lab" MainSection
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File /r "${APP_BUNDLE}/*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\Retraction Lab"
  CreateShortcut "$SMPROGRAMS\Retraction Lab\Retraction Lab.lnk" "$INSTDIR\RetractionLab.exe"
  CreateShortcut "$SMPROGRAMS\Retraction Lab\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\RetractionLab\Installer" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "DisplayName" "Retraction Lab"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "Publisher" "Retraction Lab contributors"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "DisplayIcon" "$INSTDIR\RetractionLab.exe"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab" "NoRepair" 1
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  ; Generated from the exact packaged file list. No recursive $INSTDIR removal:
  ; an unrelated file placed beside the program is preserved.
  !include "${UNINSTALL_INCLUDE}"
  Delete "$SMPROGRAMS\Retraction Lab\Retraction Lab.lnk"
  Delete "$SMPROGRAMS\Retraction Lab\Uninstall.lnk"
  RMDir "$SMPROGRAMS\Retraction Lab"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\RetractionLab"
  DeleteRegKey HKCU "Software\RetractionLab\Installer"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
SectionEnd
