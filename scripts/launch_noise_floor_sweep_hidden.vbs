' Runs the noise-floor sweep with NO console window (10.10: visible
' consoles on this desktop get closed; the 2026-07-17 16:15 sweep death
' was a window-CLOSE event, not a crash). Output still lands in
' logs\sweep_gpu_<date>.log via the .bat redirection.
CreateObject("Wscript.Shell").Run """" & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\launch_noise_floor_sweep.bat""", 0, False
