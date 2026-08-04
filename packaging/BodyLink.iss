#ifndef MyAppVersion
  #define MyAppVersion "0.3.0"
#endif
#ifndef ModelPackSHA256
  #error ModelPackSHA256 must be supplied by scripts/build_release.ps1
#endif
#ifndef ModelPackSize
  #error ModelPackSize must be supplied by scripts/build_release.ps1
#endif

#define MyAppName "BodyLink"
#define MyAppPublisher "Bytehaven-dev"
#define MyAppURL "https://github.com/Bytehaven-dev/BodyLink"
#define MyAppExeName "BodyLink.exe"
#define ModelPackName "BodyLink-Models-v" + MyAppVersion + ".zip"
#define ModelPackURL MyAppURL + "/releases/download/v" + MyAppVersion + "/" + ModelPackName

[Setup]
AppId={{C03D5726-63DD-4C11-9557-1CA69D514676}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\release
OutputBaseFilename=BodyLink-Setup-v{#MyAppVersion}
SetupIconFile=..\build\bodylink.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchiveExtraction=full
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=BodyLink model-downloading installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
MinVersion=10.0.17763

[Languages]
Name: "chinesesimp"; MessagesFile: "ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："; Flags: unchecked

[Files]
Source: "..\dist\BodyLink\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\ROADMAP.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{tmp}\{#ModelPackName}"; DestDir: "{app}\_internal"; \
  Hash: "{#ModelPackSHA256}"; ExternalSize: {#ModelPackSize}; \
  Flags: external extractarchive recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
var
  DownloadPage: TDownloadWizardPage;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(
    SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
  DownloadPage.ShowBaseNameInsteadOfUrl := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Error: String;
  LocalModelPack: String;
  TemporaryModelPack: String;
begin
  Result := True;
  if CurPageID <> wpReady then
    Exit;

  LocalModelPack := ExpandConstant('{src}\{#ModelPackName}');
  TemporaryModelPack := ExpandConstant('{tmp}\{#ModelPackName}');
  if FileExists(LocalModelPack) then
  begin
    if not CopyFile(LocalModelPack, TemporaryModelPack, False) then
    begin
      SuppressibleMsgBox(
        '无法读取安装器旁边的离线模型包：' + LocalModelPack,
        mbCriticalError, MB_OK, IDOK);
      Result := False;
    end;
    Exit;
  end;

  DownloadPage.Clear;
  DownloadPage.Add(
    '{#ModelPackURL}', '{#ModelPackName}', '{#ModelPackSHA256}');
  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
    except
      if DownloadPage.AbortedByUser then
        Log('Model pack download aborted by user.')
      else
      begin
        Error := Format('%s: %s', [
          DownloadPage.LastBaseNameOrUrl, GetExceptionMessage]);
        SuppressibleMsgBox(AddPeriod(Error), mbCriticalError, MB_OK, IDOK);
      end;
      Result := False;
    end;
  finally
    DownloadPage.Hide;
  end;
end;
