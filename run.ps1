# ==============================================================================
# Desktop Python Application Launcher
# ==============================================================================
# 使用说明：右键 -> 使用 PowerShell 运行，或在 PowerShell 中执行
# ==============================================================================

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# ------------------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------------------
$ScriptDir      = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EmbeddedEnv    = Join-Path $ScriptDir "embedded_env"
$CacheDir       = Join-Path $ScriptDir "cache"
$HFHome         = Join-Path $CacheDir  "huggingface"
$PipCacheDir    = Join-Path $CacheDir  "pip"
$SrcMain        = Join-Path $ScriptDir "src\main.py"
$PythonExe      = Join-Path $EmbeddedEnv "python.exe"
$PipExe         = Join-Path $EmbeddedEnv "Scripts\pip.exe"
$BinExe         = Join-Path $ScriptDir "bin"
$RequirementsFile = Join-Path $ScriptDir "requirements.txt"

# ------------------------------------------------------------------------------
# GitHub 更新配置（请替换为实际仓库信息）
# ------------------------------------------------------------------------------
$GitHubOwner    = "YOUR_GITHUB_USERNAME"
$GitHubRepo     = "YOUR_REPO_NAME"
$GitHubApiUrl   = "https://api.github.com/repos/$GitHubOwner/$GitHubRepo/releases/latest"
# 版本文件：本地存储当前版本号，例如内容为 "v1.0.0"
$LocalVersionFile = Join-Path $ScriptDir "version.txt"
# Python 版本信息文件（在仓库 Release 资产中，包含嵌入式 Python 版本号）
$PythonVersionFile = Join-Path $ScriptDir "python_version.txt"

# ------------------------------------------------------------------------------
# 嵌入式 Python 下载配置
# ------------------------------------------------------------------------------
$EmbedPythonVersion = "3.11.9"
$EmbedPythonUrl     = "https://www.python.org/ftp/python/$EmbedPythonVersion/python-$EmbedPythonVersion-embed-amd64.zip"
$GetPipUrl          = "https://bootstrap.pypa.io/get-pip.py"

# ==============================================================================
# 辅助函数
# ==============================================================================

function Ensure-Directories {
    foreach ($dir in @($CacheDir, $HFHome, $PipCacheDir)) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
}

function Set-AppEnvVars {
    $env:HF_HOME      = $HFHome
    $env:PIP_CACHE_DIR = $PipCacheDir
    $env:Path = "$BinExe;$env:Path"
}

function Write-Log {
    param([System.Windows.Forms.RichTextBox]$Box, [string]$Message, [string]$Color = "White")
    $Box.SelectionStart  = $Box.TextLength
    $Box.SelectionLength = 0
    $Box.SelectionColor  = [System.Drawing.ColorTranslator]::FromHtml($Color)
    $Box.AppendText("$Message`n")
    $Box.ScrollToCaret()
    $Box.Refresh()
}

# ==============================================================================
# 主窗口
# ==============================================================================

$Form = New-Object System.Windows.Forms.Form
$Form.Text            = "Application Launcher"
$Form.Size            = New-Object System.Drawing.Size(680, 560)
$Form.StartPosition   = "CenterScreen"
$Form.BackColor       = [System.Drawing.ColorTranslator]::FromHtml("#0f1117")
$Form.ForeColor       = [System.Drawing.Color]::White
$Form.FormBorderStyle = "FixedSingle"
$Form.MaximizeBox     = $false
$Form.Font            = New-Object System.Drawing.Font("Consolas", 9)

# ---------- 标题标签 ----------
$LabelTitle = New-Object System.Windows.Forms.Label
$LabelTitle.Text      = "◆  VoxCPM Studio Launcher"
$LabelTitle.Font      = New-Object System.Drawing.Font("Consolas", 13, [System.Drawing.FontStyle]::Bold)
$LabelTitle.ForeColor = [System.Drawing.ColorTranslator]::FromHtml("#00d4ff")
$LabelTitle.Location  = New-Object System.Drawing.Point(20, 16)
$LabelTitle.Size      = New-Object System.Drawing.Size(620, 28)
$Form.Controls.Add($LabelTitle)

# ---------- 分隔线 ----------
$Separator = New-Object System.Windows.Forms.Label
$Separator.BorderStyle = "Fixed3D"
$Separator.Location    = New-Object System.Drawing.Point(20, 50)
$Separator.Size        = New-Object System.Drawing.Size(625, 2)
$Form.Controls.Add($Separator)

# ---------- 按钮面板 ----------
$BtnPanel = New-Object System.Windows.Forms.Panel
$BtnPanel.Location  = New-Object System.Drawing.Point(20, 62)
$BtnPanel.Size      = New-Object System.Drawing.Size(625, 56)
$BtnPanel.BackColor = [System.Drawing.ColorTranslator]::FromHtml("#0f1117")
$Form.Controls.Add($BtnPanel)

function New-StyledButton {
    param([string]$Text, [int]$X, [string]$AccentColor)
    $btn = New-Object System.Windows.Forms.Button
    $btn.Text      = $Text
    $btn.Location  = New-Object System.Drawing.Point($X, 8)
    $btn.Size      = New-Object System.Drawing.Size(148, 38)
    $btn.FlatStyle = "Flat"
    $btn.FlatAppearance.BorderSize  = 1
    $btn.FlatAppearance.BorderColor = [System.Drawing.ColorTranslator]::FromHtml($AccentColor)
    $btn.BackColor = [System.Drawing.ColorTranslator]::FromHtml("#1a1d27")
    $btn.ForeColor = [System.Drawing.ColorTranslator]::FromHtml($AccentColor)
    $btn.Font      = New-Object System.Drawing.Font("Consolas", 8, [System.Drawing.FontStyle]::Bold)
    $btn.Cursor    = [System.Windows.Forms.Cursors]::Hand
    return $btn
}

$BtnLaunch  = New-StyledButton "▶  启动主程序"   0   "#00ff88"
$BtnInit    = New-StyledButton "⚙  初始化环境"   152 "#00d4ff"
$BtnUpdate  = New-StyledButton "↑  检查更新"     304 "#ffaa00"
$BtnSysInfo = New-StyledButton "◉  系统信息"     456 "#cc88ff"

$BtnPanel.Controls.AddRange(@($BtnLaunch, $BtnInit, $BtnUpdate, $BtnSysInfo))

# ---------- 状态标签 ----------
$StatusLabel = New-Object System.Windows.Forms.Label
$StatusLabel.Text      = "就绪"
$StatusLabel.Location  = New-Object System.Drawing.Point(20, 124)
$StatusLabel.Size      = New-Object System.Drawing.Size(625, 18)
$StatusLabel.ForeColor = [System.Drawing.ColorTranslator]::FromHtml("#888888")
$StatusLabel.Font      = New-Object System.Drawing.Font("Consolas", 8)
$Form.Controls.Add($StatusLabel)

# ---------- 日志输出框 ----------
$LogBox = New-Object System.Windows.Forms.RichTextBox
$LogBox.Location   = New-Object System.Drawing.Point(20, 148)
$LogBox.Size       = New-Object System.Drawing.Size(625, 350)
$LogBox.BackColor  = [System.Drawing.ColorTranslator]::FromHtml("#080a0f")
$LogBox.ForeColor  = [System.Drawing.Color]::White
$LogBox.Font       = New-Object System.Drawing.Font("Consolas", 9)
$LogBox.ReadOnly   = $true
$LogBox.BorderStyle = "None"
$LogBox.ScrollBars  = "Vertical"
$Form.Controls.Add($LogBox)

# ---------- 清空日志按钮 ----------
$BtnClear = New-Object System.Windows.Forms.Button
$BtnClear.Text     = "清空"
$BtnClear.Location = New-Object System.Drawing.Point(590, 120)
$BtnClear.Size     = New-Object System.Drawing.Size(55, 22)
$BtnClear.FlatStyle = "Flat"
$BtnClear.FlatAppearance.BorderColor = [System.Drawing.ColorTranslator]::FromHtml("#333344")
$BtnClear.BackColor = [System.Drawing.ColorTranslator]::FromHtml("#1a1d27")
$BtnClear.ForeColor = [System.Drawing.ColorTranslator]::FromHtml("#888888")
$BtnClear.Font     = New-Object System.Drawing.Font("Consolas", 7)
$BtnClear.Cursor   = [System.Windows.Forms.Cursors]::Hand
$BtnClear.Add_Click({ $LogBox.Clear() })
$Form.Controls.Add($BtnClear)

# ------------------------------------------------------------------------------
# 全局变量 — 主程序进程追踪
# ------------------------------------------------------------------------------
$script:mainProcess = $null

# 初始化目录和环境变量
Ensure-Directories
Set-AppEnvVars

Write-Log $LogBox "═══════════════════════════════════════════════" "#333355"
Write-Log $LogBox "  Python App Launcher 已就绪" "#00d4ff"
Write-Log $LogBox "  根目录：$ScriptDir" "#666688"
Write-Log $LogBox "  缓存目录：$CacheDir" "#666688"
Write-Log $LogBox "═══════════════════════════════════════════════" "#333355"

# ------------------------------------------------------------------------------
# 辅助函数 — 切换按钮启动/停止状态
# ------------------------------------------------------------------------------
function Update-ProcessButtonState {
    param([bool]$IsRunning)
    if ($IsRunning) {
        $BtnLaunch.Text       = "[infinity]  停止主程序"
        $BtnLaunch.ForeColor   = [System.Drawing.ColorTranslator]::FromHtml("#ff4466")
        $BtnLaunch.BackColor  = [System.Drawing.ColorTranslator]::FromHtml("#2a1520")
        $BtnLaunch.FlatAppearance.BorderColor = [System.Drawing.ColorTranslator]::FromHtml("#ff4466")
    } else {
        $BtnLaunch.Text       = "▶  启动主程序"
        $BtnLaunch.ForeColor   = [System.Drawing.ColorTranslator]::FromHtml("#00ff88")
        $BtnLaunch.BackColor  = [System.Drawing.ColorTranslator]::FromHtml("#1a1d27")
        $BtnLaunch.FlatAppearance.BorderColor = [System.Drawing.ColorTranslator]::FromHtml("#00ff88")
    }
}

# ==============================================================================
# 按钮 1：启动主程序
# ==============================================================================
$BtnLaunch.Add_Click({
    $isRunning = ($null -ne $script:mainProcess) -and (-not $script:mainProcess.HasExited)

    if ($isRunning) {
        # ── 停止主程序 ──
        Write-Log $LogBox "`n[停止] ──────────────────────────────" "#ff4466"
        $StatusLabel.Text = "正在停止主程序..."
        [System.Windows.Forms.Application]::DoEvents()

        $processToStop = $script:mainProcess
        $script:mainProcess = $null

        try {
            Stop-Process -Id $processToStop.Id -Force -ErrorAction SilentlyContinue
            Write-Log $LogBox "[成功] 主程序已终止（PID: $($processToStop.Id)）。" "#ff4466"
            $StatusLabel.Text = "主程序已停止"
        } catch {
            Write-Log $LogBox "[错误] 终止失败：$($_.Exception.Message)" "#ff4466"
            $StatusLabel.Text = "终止失败"
        }

        Update-ProcessButtonState -IsRunning $false
        return
    }

    # ── 启动主程序 ──
    $StatusLabel.Text = "正在启动主程序..."
    Write-Log $LogBox "`n[启动] ──────────────────────────────" "#00ff88"
    [System.Windows.Forms.Application]::DoEvents()

    if (-not (Test-Path $EmbeddedEnv)) {
        Write-Log $LogBox "[错误] embedded_env 目录不存在，请先初始化环境。" "#ff4466"
        $StatusLabel.Text = "错误：embedded_env 不存在"
        [System.Windows.Forms.MessageBox]::Show(
            "嵌入式 Python 环境不存在。`n请先点击「初始化环境」。",
            "启动失败",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
        return
    }

    if (-not (Test-Path $PythonExe)) {
        Write-Log $LogBox "[错误] 未找到 python.exe：$PythonExe" "#ff4466"
        $StatusLabel.Text = "错误：python.exe 不存在"
        return
    }

    if (-not (Test-Path $SrcMain)) {
        Write-Log $LogBox "[错误] 未找到主程序文件：$SrcMain" "#ff4466"
        $StatusLabel.Text = "错误：src\main.py 不存在"
        return
    }

    Set-AppEnvVars
    Write-Log $LogBox "[信息] HF_HOME      = $env:HF_HOME" "#888888"
    Write-Log $LogBox "[信息] PIP_CACHE_DIR = $env:PIP_CACHE_DIR" "#888888"
    Write-Log $LogBox "[运行] $PythonExe $SrcMain" "#00ff88"

    try {
        $script:mainProcess = Start-Process -FilePath $PythonExe -ArgumentList "`"$SrcMain`" " `
            -WorkingDirectory $ScriptDir -PassThru -NoNewWindow -ErrorAction SilentlyContinue
        if ($null -ne $script:mainProcess) {
            Write-Log $LogBox "[成功] 主程序已启动（PID: $($script:mainProcess.Id)）。" "#00ff88"
            Update-ProcessButtonState -IsRunning $true
        } else {
            throw "Start-Process returned null"
        }
        $StatusLabel.Text = "主程序已启动（PID: $($script:mainProcess.Id)）"
    } catch {
        Write-Log $LogBox "[错误] 启动失败：$($_.Exception.Message)" "#ff4466"
        $StatusLabel.Text = "启动失败"
    }
})

# ==============================================================================
# 按钮 2：初始化环境
# ==============================================================================
$BtnInit.Add_Click({
    $StatusLabel.Text = "正在初始化环境..."
    Write-Log $LogBox "`n[初始化] ────────────────────────────" "#00d4ff"
    [System.Windows.Forms.Application]::DoEvents()

    # 下载嵌入式 Python
    $ZipPath = Join-Path $env:TEMP "python-embed.zip"
    Write-Log $LogBox "[下载] $EmbedPythonUrl" "#888888"
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($EmbedPythonUrl, $ZipPath)
        Write-Log $LogBox "[成功] Python 压缩包下载完成。" "#00d4ff"
    } catch {
        Write-Log $LogBox "[错误] 下载失败：$($_.Exception.Message)" "#ff4466"
        $StatusLabel.Text = "下载失败"
        return
    }

    # 解压到 embedded_env
    if (Test-Path $EmbeddedEnv) {
        Write-Log $LogBox "[信息] 清理旧的 embedded_env..." "#888888"
        Remove-Item -Recurse -Force $EmbeddedEnv
    }
    Write-Log $LogBox "[解压] 解压到 $EmbeddedEnv ..." "#888888"
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $EmbeddedEnv)
        Write-Log $LogBox "[成功] Python 解压完成。" "#00d4ff"
    } catch {
        Write-Log $LogBox "[错误] 解压失败：$($_.Exception.Message)" "#ff4466"
        $StatusLabel.Text = "解压失败"
        return
    }

    # 修改 ._pth 文件以启用 site-packages（嵌入式 Python 默认禁用）
    $PthFile = Get-ChildItem -Path $EmbeddedEnv -Filter "*._pth" | Select-Object -First 1
    if ($PthFile) {
        $PthContent = Get-Content $PthFile.FullName -Raw
        $PthContent = $PthContent -replace "#import site", "import site"
        Set-Content $PthFile.FullName $PthContent
        Write-Log $LogBox "[配置] 已启用 site-packages（$($PthFile.Name)）" "#888888"
    }

    # 下载 get-pip.py
    $GetPipPath = Join-Path $env:TEMP "get-pip.py"
    Write-Log $LogBox "[下载] get-pip.py ..." "#888888"
    try {
        $wc.DownloadFile($GetPipUrl, $GetPipPath)
        Write-Log $LogBox "[成功] get-pip.py 下载完成。" "#00d4ff"
    } catch {
        Write-Log $LogBox "[错误] get-pip.py 下载失败：$($_.Exception.Message)" "#ff4466"
        $StatusLabel.Text = "get-pip 下载失败"
        return
    }

    # 安装 pip
    Write-Log $LogBox "[安装] 正在安装 pip..." "#888888"
    try {
        $proc = Start-Process -FilePath $PythonExe -ArgumentList "`"$GetPipPath`"" `
            -WorkingDirectory $EmbeddedEnv -Wait -PassThru -NoNewWindow
        if ($proc.ExitCode -ne 0) {
            Write-Log $LogBox "[错误] pip 安装失败，退出码：$($proc.ExitCode)" "#ff4466"
            $StatusLabel.Text = "pip 安装失败"
            return
        }
        Write-Log $LogBox "[成功] pip 安装完成。" "#00d4ff"
    } catch {
        Write-Log $LogBox "[错误] $($_.Exception.Message)" "#ff4466"
        $StatusLabel.Text = "pip 安装异常"
        return
    }

    # 安装第三方依赖
    if (Test-Path $RequirementsFile) {
        Write-Log $LogBox "[安装] 正在安装 requirements.txt 依赖..." "#888888"
        $PipArgs = "-m pip install -r `"$RequirementsFile`" --cache-dir `"$PipCacheDir`""
        try {
            $proc = Start-Process -FilePath $PythonExe -ArgumentList $PipArgs `
                -WorkingDirectory $ScriptDir -Wait -PassThru -NoNewWindow
            if ($proc.ExitCode -ne 0) {
                Write-Log $LogBox "[警告] 依赖安装退出码：$($proc.ExitCode)，请检查输出。" "#ffaa00"
            } else {
                Write-Log $LogBox "[成功] 所有依赖安装完成。" "#00d4ff"
            }
        } catch {
            Write-Log $LogBox "[错误] $($_.Exception.Message)" "#ff4466"
        }
    } else {
        Write-Log $LogBox "[跳过] 未找到 requirements.txt，跳过依赖安装。" "#888888"
    }

    # 记录当前 Python 版本
    $EmbedPythonVersion | Set-Content $PythonVersionFile

    Write-Log $LogBox "──────────────────────────────────────" "#333355"
    Write-Log $LogBox "[完成] 环境初始化成功！✓" "#00ff88"
    $StatusLabel.Text = "环境初始化完成"

    [System.Windows.Forms.MessageBox]::Show(
        "嵌入式 Python 环境初始化成功！`nPython 版本：$EmbedPythonVersion",
        "初始化完成",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
})

# ==============================================================================
# 按钮 3：检查更新
# ==============================================================================
$BtnUpdate.Add_Click({
    $StatusLabel.Text = "正在检查更新..."
    Write-Log $LogBox "`n[更新] ──────────────────────────────" "#ffaa00"
    [System.Windows.Forms.Application]::DoEvents()

    # 读取本地版本号
    $LocalVersion = "v0.0.0"
    if (Test-Path $LocalVersionFile) {
        $LocalVersion = (Get-Content $LocalVersionFile -Raw).Trim()
    }
    Write-Log $LogBox "[信息] 本地版本：$LocalVersion" "#888888"

    # 查询 GitHub API
    Write-Log $LogBox "[网络] 查询 $GitHubApiUrl ..." "#888888"
    try {
        $Headers = @{ "User-Agent" = "PyAppLauncher/1.0" }
        $Response = Invoke-RestMethod -Uri $GitHubApiUrl -Headers $Headers -ErrorAction Stop
        $LatestTag = $Response.tag_name
        $ReleaseNotes = $Response.body
        Write-Log $LogBox "[信息] 最新版本：$LatestTag" "#ffaa00"
    } catch {
        Write-Log $LogBox "[错误] 无法访问 GitHub API：$($_.Exception.Message)" "#ff4466"
        Write-Log $LogBox "[提示] 请确认仓库地址配置正确，或检查网络连接。" "#888888"
        $StatusLabel.Text = "检查更新失败"
        return
    }

    if ($LatestTag -eq $LocalVersion) {
        Write-Log $LogBox "[完成] 已是最新版本 $LocalVersion，无需更新。✓" "#00ff88"
        $StatusLabel.Text = "已是最新版本"
        [System.Windows.Forms.MessageBox]::Show(
            "当前已是最新版本：$LocalVersion",
            "无需更新",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        return
    }

    # 发现新版本，询问用户
    $Msg = "发现新版本：$LatestTag`n当前版本：$LocalVersion`n`n更新说明：`n$ReleaseNotes`n`n是否立即更新？"
    $Confirm = [System.Windows.Forms.MessageBox]::Show(
        $Msg, "发现新版本",
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Question
    )
    if ($Confirm -ne [System.Windows.Forms.DialogResult]::Yes) {
        Write-Log $LogBox "[取消] 用户取消更新。" "#888888"
        $StatusLabel.Text = "已取消更新"
        return
    }

    # 下载 Release 资产（假设主资产为 app.zip，可根据实际修改）
    $Asset = $Response.assets | Where-Object { $_.name -like "*.zip" } | Select-Object -First 1
    if (-not $Asset) {
        Write-Log $LogBox "[错误] Release 中未找到可下载的 .zip 资产。" "#ff4466"
        $StatusLabel.Text = "未找到更新包"
        return
    }

    $DownloadUrl  = $Asset.browser_download_url
    $DownloadPath = Join-Path $env:TEMP $Asset.name
    Write-Log $LogBox "[下载] $DownloadUrl" "#888888"
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($DownloadUrl, $DownloadPath)
        Write-Log $LogBox "[成功] 下载完成：$($Asset.name)" "#ffaa00"
    } catch {
        Write-Log $LogBox "[错误] 下载失败：$($_.Exception.Message)" "#ff4466"
        $StatusLabel.Text = "下载更新失败"
        return
    }

    # 检查更新包中的 Python 版本
    $NewPythonVersion = $null
    $TempExtract = Join-Path $env:TEMP "app_update_check"
    if (Test-Path $TempExtract) { Remove-Item -Recurse -Force $TempExtract }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($DownloadPath, $TempExtract)

    $NewPyVerFile = Join-Path $TempExtract "python_version.txt"
    if (Test-Path $NewPyVerFile) {
        $NewPythonVersion = (Get-Content $NewPyVerFile -Raw).Trim()
    }

    $LocalPyVersion = $null
    if (Test-Path $PythonVersionFile) {
        $LocalPyVersion = (Get-Content $PythonVersionFile -Raw).Trim()
    }

    $PythonNeedsReinit = $false
    if ($NewPythonVersion -and ($NewPythonVersion -ne $LocalPyVersion)) {
        Write-Log $LogBox "[信息] Python 版本变更：$LocalPyVersion → $NewPythonVersion，将重新初始化环境。" "#ffaa00"
        $PythonNeedsReinit = $true
        $script:EmbedPythonVersion = $NewPythonVersion
        $script:EmbedPythonUrl = "https://www.python.org/ftp/python/$NewPythonVersion/python-$NewPythonVersion-embed-amd64.zip"
    }

    # 覆盖安装到项目根目录（保留 embedded_env 和 cache）
    Write-Log $LogBox "[解压] 正在将更新覆盖到项目目录..." "#888888"
    Get-ChildItem -Path $TempExtract | ForEach-Object {
        $Dest = Join-Path $ScriptDir $_.Name
        if ($_.Name -notin @("embedded_env", "cache")) {
            Copy-Item -Path $_.FullName -Destination $Dest -Recurse -Force
        }
    }

    # 更新本地版本号
    $LatestTag | Set-Content $LocalVersionFile
    Write-Log $LogBox "[成功] 项目文件已更新至 $LatestTag。" "#ffaa00"

    # 若需要重新初始化 Python 环境
    if ($PythonNeedsReinit) {
        Write-Log $LogBox "[信息] 移除旧的 embedded_env..." "#888888"
        if (Test-Path $EmbeddedEnv) { Remove-Item -Recurse -Force $EmbeddedEnv }
        Write-Log $LogBox "[提示] Python 环境已移除，请点击「初始化环境」重新安装。" "#ffaa00"
        [System.Windows.Forms.MessageBox]::Show(
            "更新完成！`n`nPython 版本已变更（$LocalPyVersion → $NewPythonVersion）。`n请点击「初始化环境」重新安装 Python 环境。",
            "更新完成",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
    } else {
        Write-Log $LogBox "[完成] 更新完成！✓" "#00ff88"
        [System.Windows.Forms.MessageBox]::Show(
            "更新完成！当前版本：$LatestTag",
            "更新成功",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    }

    $StatusLabel.Text = "更新完成：$LatestTag"
    Remove-Item -Recurse -Force $TempExtract -ErrorAction SilentlyContinue
})

# ==============================================================================
# 按钮 4：系统信息
# ==============================================================================
$BtnSysInfo.Add_Click({
    $StatusLabel.Text = "正在获取系统信息..."
    Write-Log $LogBox "`n[系统信息] ──────────────────────────" "#cc88ff"
    [System.Windows.Forms.Application]::DoEvents()

    # 运行 dxdiag 导出到临时文件
    $DxDiagOut = Join-Path $env:TEMP "dxdiag_output.txt"
    Write-Log $LogBox "[运行] dxdiag /t $DxDiagOut（可能需要数秒）..." "#888888"

    try {
        $proc = Start-Process -FilePath "dxdiag" -ArgumentList "/t `"$DxDiagOut`"" -Wait -PassThru -NoNewWindow
    } catch {
        Write-Log $LogBox "[错误] 无法运行 dxdiag：$($_.Exception.Message)" "#ff4466"
        $StatusLabel.Text = "dxdiag 运行失败"
        return
    }

    # 等待文件生成
    $Timeout = 60
    $Elapsed = 0
    while (-not (Test-Path $DxDiagOut) -and $Elapsed -lt $Timeout) {
        Start-Sleep -Milliseconds 500
        $Elapsed += 0.5
    }

    if (-not (Test-Path $DxDiagOut)) {
        Write-Log $LogBox "[错误] dxdiag 输出文件未生成，请重试。" "#ff4466"
        $StatusLabel.Text = "dxdiag 超时"
        return
    }

    $Content = Get-Content $DxDiagOut -Encoding Unicode -Raw

    # 提取关键信息
    function Extract-Field {
        param([string]$Text, [string]$FieldName)
        if ($Text -match "$FieldName\s*:\s*(.+)") { return $Matches[1].Trim() }
        return "未知"
    }

    $OSName       = Extract-Field $Content "Operating System"
    $ComputerName = Extract-Field $Content "Machine name"
    $Processor    = Extract-Field $Content "Processor"
    $Memory       = Extract-Field $Content "Available OS RAM"
    if ($Memory -eq "未知") { $Memory = Extract-Field $Content "Memory" }

    # 提取所有显卡
    $GPUMatches = [regex]::Matches($Content, "Card name\s*:\s*(.+)")
    $GPUs = @()
    foreach ($m in $GPUMatches) { $GPUs += $m.Groups[1].Value.Trim() }

    $VRAMMatches = [regex]::Matches($Content, "Dedicated Memory\s*:\s*(.+)")
    $VRAMs = @()
    foreach ($m in $VRAMMatches) { $VRAMs += $m.Groups[1].Value.Trim() }

    $DriverMatches = [regex]::Matches($Content, "Driver Version\s*:\s*(.+)")
    $Drivers = @()
    foreach ($m in $DriverMatches) { $Drivers += $m.Groups[1].Value.Trim() }

    # 输出信息
    Write-Log $LogBox "" ""
    Write-Log $LogBox "  ┌─────────────────────────────────┐" "#444466"
    Write-Log $LogBox "  │         系统基本信息              │" "#cc88ff"
    Write-Log $LogBox "  └─────────────────────────────────┘" "#444466"
    Write-Log $LogBox "  计算机名  ：$ComputerName" "#ddddff"
    Write-Log $LogBox "  操作系统  ：$OSName" "#ddddff"
    Write-Log $LogBox "  处理器    ：$Processor" "#ddddff"
    Write-Log $LogBox "  内存      ：$Memory" "#ddddff"

    Write-Log $LogBox "" ""
    Write-Log $LogBox "  ┌─────────────────────────────────┐" "#444466"
    Write-Log $LogBox "  │         显卡信息                  │" "#cc88ff"
    Write-Log $LogBox "  └─────────────────────────────────┘" "#444466"
    for ($i = 0; $i -lt $GPUs.Count; $i++) {
        $VramStr   = if ($i -lt $VRAMs.Count)   { $VRAMs[$i]   } else { "N/A" }
        $DriverStr = if ($i -lt $Drivers.Count) { $Drivers[$i] } else { "N/A" }
        Write-Log $LogBox "  显卡 $($i+1)   ：$($GPUs[$i])" "#ddddff"
        Write-Log $LogBox "  显存      ：$VramStr" "#ddddff"
        Write-Log $LogBox "  驱动版本  ：$DriverStr" "#ddddff"
        if ($i -lt $GPUs.Count - 1) { Write-Log $LogBox "  ·················" "#333355" }
    }

    # 补充 PowerShell 系统信息
    Write-Log $LogBox "" ""
    Write-Log $LogBox "  ┌─────────────────────────────────┐" "#444466"
    Write-Log $LogBox "  │         详细内存信息              │" "#cc88ff"
    Write-Log $LogBox "  └─────────────────────────────────┘" "#444466"
    try {
        $CS = Get-CimInstance Win32_ComputerSystem
        $TotalRAM  = [math]::Round($CS.TotalPhysicalMemory / 1GB, 2)
        $OS_obj    = Get-CimInstance Win32_OperatingSystem
        $FreeRAM   = [math]::Round($OS_obj.FreePhysicalMemory / 1MB, 2)
        $UsedRAM   = [math]::Round($TotalRAM - $FreeRAM / 1024, 2)
        Write-Log $LogBox "  总内存    ：${TotalRAM} GB" "#ddddff"
        Write-Log $LogBox "  可用内存  ：${FreeRAM} MB" "#ddddff"
    } catch {
        Write-Log $LogBox "  [跳过] 无法获取详细内存信息。" "#888888"
    }

    Write-Log $LogBox "" ""
    Write-Log $LogBox "[完成] 系统信息获取完成。✓" "#cc88ff"
    $StatusLabel.Text = "系统信息已获取"
})

# ==============================================================================
# 运行主循环
# ==============================================================================
[System.Windows.Forms.Application]::Run($Form)
