@echo off
echo ===================================
echo RESET COMPLETO RAG CHATBOT
echo ===================================
echo.
echo ATTENZIONE: Questa operazione eliminerà DEFINITIVAMENTE:
echo - Tutti i chatbot creati
echo - Tutti i documenti caricati
echo - Tutti i database vettoriali
echo - Tutte le personalizzazioni (template, impostazioni)
echo.
echo L'applicazione tornerà allo stato iniziale "vergine" e
echo tutti i dati verranno persi in modo IRREVERSIBILE.
echo.

set /p CONFIRM=Sei sicuro di voler procedere? (S/N): 

if /i "%CONFIRM%" neq "S" (
    echo.
    echo Operazione annullata.
    goto :end
)

echo.
echo Eliminazione dei dati in corso...

REM Elimina la directory dei chatbot e tutto il suo contenuto
if exist "chatbots\" (
    echo - Eliminazione di tutti i chatbot...
    rmdir /s /q "chatbots"
)

REM Ricrea la directory dei chatbot vuota
mkdir "chatbots"

REM Elimina eventuali file temporanei
echo - Pulizia dei file temporanei...
if exist "temp\" (
    rmdir /s /q "temp"
)

REM Elimina cache di Streamlit
echo - Pulizia della cache di Streamlit...
if exist ".streamlit\" (
    rmdir /s /q ".streamlit"
)

echo.
echo ===================================
echo RESET COMPLETATO CON SUCCESSO
echo ===================================
echo.
echo L'applicazione è stata ripristinata allo stato iniziale.
echo Tutti i chatbot, documenti e personalizzazioni sono stati eliminati.
echo.
echo Al prossimo avvio, l'applicazione sarà come nuova.
echo.

:end
pause 