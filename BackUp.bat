@echo off
SETLOCAL EnableDelayedExpansion
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "fullstamp=%MM%_%DD%_%YYYY%_%HH%%Min%%Sec%"

set remoteBackupDir=E:\Sentiment_Backup
set newBackupName=\SentBackup_%fullstamp%\
set srcDir=C:\Users\local_uc4u76k\Documents\Programming\MachineLearning\News_Sentiment
set "finalMsg="
set useRemote=false
IF EXIST %remoteBackupDir% SET useRemote=true

IF "%useRemote%"=="true" (
xcopy /s %srcDir% %remoteBackupDir%%newBackupName%

    set "finalMsg=Remote backup completed: %remoteBackupDir%%newBackupName%"
) ELSE (

 set "finalMsg=No remote drive found, no backup made"

)

echo !finalMsg!
pause
