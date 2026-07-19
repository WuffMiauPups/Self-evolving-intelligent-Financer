' Runs the NEWSBOX N0 collector (backfill once, then poll forever) with NO
' console window (MASTER_PLAN 10 note 10). Output: logs\news_backfill.log /
' logs\news_poller.log, milestones in logs\news_collector.status. Detached
' (False) so it survives session restarts; only a shutdown ends it.
CreateObject("Wscript.Shell").Run """" & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\launch_news_collector.bat""", 0, False
