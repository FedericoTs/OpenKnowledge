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
; Inno replaces files. It never removes ones the new build no longer has, so
; without this every file any past version ever shipped accumulates in the
; runtime directory for ever.
;
; That is not theoretical. PyInstaller bundles the package's own .dist-info so
; the frozen app can read its version, and the directory name carries that
; version. Measured on a real 0.2.18 -> 0.2.19 upgrade in CI: every file was
; replaced, the installer exited 0, and the installed app still reported
; 0.2.18, because importlib.metadata answered out of whichever of the two
; directories it found first. The updater then compared each new release
; against that stale number and offered the same update again, for ever. This
; is why an install could never be seen to update itself.
;
; Deleting only that one directory fixed the defect that was measured and left
; the class of defect open: any module, DLL or data file a future build stops
; shipping stays on disk, where it can shadow the new one or simply mislead.
; So the whole runtime goes, and [Files] lays down exactly what this build
; carries. Nothing here is the person's: documents, database, models and
; settings live under {localappdata}\OpenKnowledge and are never touched.
;
; Safe only because PrepareToInstall has already stopped anything holding
; these files - see [Code]. If a file is still locked, Inno logs it and
; carries on, which leaves the old behaviour rather than a broken install.
Type: filesandordirs; Name: "{app}\_internal"

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
{ Stop anything holding the runtime, so it can be replaced wholesale.

  The app's own updater already waits for its process to exit before running
  this, so on that path there is nothing to stop. This is for the other way
  an upgrade happens: somebody double-clicks the installer with the app open.
  Without it the wholesale delete above would silently do nothing on exactly
  the files it exists to remove.

  taskkill filters by image name, which is machine-wide - somebody running
  their own llama-server.exe while upgrading OpenKnowledge would lose it too.
  The same trade the uninstaller already makes, and for the same reason: the
  process is restartable, and files held by a live one would otherwise
  survive. Exit codes are ignored on purpose; 128 means "not running", which
  is the normal case and not a problem. }
procedure StopTheRunningApp();
var
  Names: array[0..2] of String;
  I, ResultCode: Integer;
begin
  Names[0] := 'OpenKnowledgeApp.exe';
  Names[1] := 'openknowledge.exe';
  Names[2] := 'llama-server.exe';
  for I := 0 to 2 do
    Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM ' + Names[I], '',
         SW_HIDE, ewWaitUntilTerminated, ResultCode);
  { Killing a process and Windows releasing its file handles are not the same
    instant, and the delete that follows is the whole point of stopping it. }
  Sleep(750);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopTheRunningApp();
  Result := '';
end;

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
