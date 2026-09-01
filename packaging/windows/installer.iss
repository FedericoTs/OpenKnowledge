; OpenKnowledge Windows installer.
;
; Per-user by design: PrivilegesRequired=lowest installs under
; %LOCALAPPDATA%\Programs with no UAC prompt, which is both the least
; privilege the app needs and the honest shape for an unsigned installer -
; asking for administrator rights without a signature is how installers
; train people to click through warnings.
;
; Build (CI does this; see .github/workflows/package.yml):
;   uv run pyinstaller packaging/pyinstaller/openknowledge.spec
;   powershell packaging/windows/fetch-llama.ps1
;   iscc /DAppVersion=x.y.z packaging/windows/installer.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{C7E64A2B-9A0D-4E1F-8B36-5D2A19F0C4E7}
AppName=OpenKnowledge
AppVersion={#AppVersion}
AppPublisher=OpenKnowledge contributors
AppPublisherURL=https://github.com/FedericoTs/OpenKnowledge
DefaultDirName={autopf}\OpenKnowledge
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename=OpenKnowledge-Setup-{#AppVersion}
SetupIconFile=openknowledge.ico
UninstallDisplayIcon={app}\OpenKnowledgeApp.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\..\LICENSE
InfoBeforeFile=before-install.txt
ChangesEnvironment=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked
Name: "addtopath"; Description: "Add the &command line tool to PATH (for `openknowledge ask`, audits, scripting)"; Flags: unchecked

[InstallDelete]
; PyInstaller bundles the package's own .dist-info so the frozen app can read
; its version, and the directory name carries that version. Inno replaces
; files; it never removes ones the new build no longer has - so an upgrade
; left both, and importlib.metadata answered with whichever it found first.
;
; Measured on a real 0.2.18 -> 0.2.19 upgrade in CI: every file was replaced,
; the installer exited 0, and the installed app still reported 0.2.18. The
; updater then compared each new release against that stale number and
; offered the same update again, for ever. This is why an install could
; never be seen to update itself.
;
; Scoped to the metadata rather than all of {app}\_internal: this is the
; defect that is measured, and a wholesale delete of the runtime would fail
; badly if someone ran the installer while the app still held those files.
; Anything else the new build drops still lingers - see ROADMAP.
Type: filesandordirs; Name: "{app}\_internal\openknowledge-*.dist-info"

[Files]
; The PyInstaller onedir bundle: both executables and their shared runtime.
Source: "..\..\dist\OpenKnowledge\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs
; The llama.cpp server, where find_llama_server() looks: {app}\llama.
Source: "..\..\build\llama\*"; DestDir: "{app}\llama"; Flags: ignoreversion recursesubdirs
Source: "..\..\build\llama-version.txt"; DestDir: "{app}\llama"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\OpenKnowledge"; Filename: "{app}\OpenKnowledgeApp.exe"
Name: "{autodesktop}\OpenKnowledge"; Filename: "{app}\OpenKnowledgeApp.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\OpenKnowledgeApp.exe"; Description: "Start OpenKnowledge now"; \
  Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop what we started before removing files. taskkill filters by image
; name, which is machine-wide: someone running their own llama-server.exe
; while uninstalling OpenKnowledge would lose it too. Accepted - the overlap
; is rare, the process restartable, and files locked by a live process would
; otherwise survive the uninstall.
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM OpenKnowledgeApp.exe"; Flags: runhidden; RunOnceId: "KillApp"
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM llama-server.exe"; Flags: runhidden; RunOnceId: "KillLlama"

[UninstallDelete]
; The install folder only. The person's state - documents, database, models,
; the .env with their settings - lives under {localappdata}\OpenKnowledge and
; is deliberately NOT deleted: uninstalling an app should not destroy the
; knowledge base someone built with it.
Type: filesandordirs; Name: "{app}"

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  { look for the dir, delimited on both sides, in the existing PATH }
  Result := Pos(';' + Uppercase(Param) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;
