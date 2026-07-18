' Runs the arm-3 horizon-extension batch with NO console window (MASTER_PLAN
' 10 note 10: visible consoles on this desktop get closed; the 2026-07-17
' 16:15 sweep death was a window-CLOSE event, not a crash). Output lands in
' logs\horizon_arm3_h15.log / _h30.log and milestones in logs\horizon_arm3.status
' via the .bat redirection. Detached (False) so it survives session restarts.
CreateObject("Wscript.Shell").Run """" & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\launch_horizon_arm3.bat""", 0, False
