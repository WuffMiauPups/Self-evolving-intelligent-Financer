' Runs the arm-4 risk-target-norm batch with NO console window (MASTER_PLAN
' 10 note 10: visible consoles on this desktop get closed). Output lands in
' logs\risk_norm_arm4.log and milestones in logs\risk_norm_arm4.status via the
' .bat redirection. Detached (False) so it survives session restarts.
CreateObject("Wscript.Shell").Run """" & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\launch_risk_norm_arm4.bat""", 0, False
