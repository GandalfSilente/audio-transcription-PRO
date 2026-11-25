# Trascrizione Audio Locale (GTK 4 + Libadwaita + faster-whisper)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE-MIT) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE-APACHE)

Applicazione desktop nativa per Linux (con script CLI incluso) per trascrivere file audio e video lunghi **localmente** con `faster-whisper`, accelerazione **GPU NVIDIA CUDA** (e fallback automatico su CPU) e filtro vocale **VAD**. Dopo aver scaricato i modelli necessari, la trascrizione può funzionare offline.

Progettata per avviarsi istantaneamente senza browser né server web locali, con supporto a Drag & Drop nativo Wayland, coda multi-file e profili di contesto personalizzabili per adattare il riconoscimento a qualsiasi disciplina (medicina, fisica, informatica, ecc.).

---

## Funzionalità principali

- **Interfaccia desktop nativa Linux (`GTK 4 + Libadwaita`)**: tema scuro leggero, avvio istantaneo tramite *lazy loading* delle librerie CUDA e di `faster-whisper` in background (0 VRAM occupata all'avvio).
- **Drag & Drop multi-file e coda sequenziale**: trascina uno o più file audio/video (o usa il pulsante *Sfoglia file...*); eventuali file aggiunti durante una trascrizione vengono messi in coda automaticamente.
- **4 modelli di trascrizione**:
  - `large-v3` *(predefinito)*: massima accuratezza.
  - `Combinato (Ensemble: large-v3-turbo + large-v3)`: doppio passaggio con confronto sulla confidenza (`avg_logprob`) per migliorare i segmenti incerti mantenendo basso il picco di VRAM.
  - `large-v3-turbo`: ultra veloce, ideale per audio molto lunghi o PC meno potenti.
  - `Parakeet TDT 0.6B v3`: modello NVIDIA via ONNX, vedi sotto.
- **Interruttore PRO (LLM locale)**: dopo la trascrizione, corregge errori e migliora la leggibilità con LM Studio o Ollama. Attivando PRO, cerca tutti i modelli chat installati; se ne trova più di uno, puoi scegliere. I servizi e il modello partono in background, senza aprire le app. Disattivando PRO, l'app scarica il modello e chiude i servizi avviati da essa; quelli già attivi prima restano aperti. Il pulsante **Libera memoria** fa lo stesso senza attendere la fine dell'app. Riattiva l'interruttore per caricare di nuovo il modello. L'originale `.txt` resta invariato e il risultato viene salvato in `.pro.txt`.
- **Profili di contesto personalizzabili**: menu a tendina con preset pronti (*Generale*, *Medicina e Radiologia IT*, *Fisica / Fisica Nucleare EN*, *Fisica e Ingegneria IT*, *Informatica e AI EN/IT*) più un pannello espandibile per scrivere/modificare il prompt tecnico e salvare o eliminare i propri profili personalizzati in `~/.config/trascrizione-audio/config.json` (fuori dal repository Git). Il contesto guida sia il riconoscimento sia PRO nell'interpretazione dei termini; PRO non forza correzioni se frase e contesto non le confermano. Ogni profilo imposta automaticamente la lingua consigliata (sempre sovrascrivibile manualmente) e l'app ricorda le ultime scelte tra un avvio e l'altro.
- **Rilevamento lingua e VAD calibrato per aule**: selezione lingua (`Automatica IT/EN`, `Italiano`, `Inglese`) e Silero VAD ottimizzato per microfoni d'aula/distanti (`threshold=0.35`, `min_silence_duration_ms=600`, `speech_pad_ms=400`) per evitare tagli delle parole iniziali e ridurre le allucinazioni nei silenzi.
- **Riconoscimento voci (Speaker Diarization ONNX)**: menu a tendina **Riconoscimento voci** (`NO` predefinito, `SI` per più voci sicure, `Auto` per rilevare automaticamente se parla più di una persona). Assegna `[Voce 1]`, `[Voce 2]`, ... in ordine di apparizione usando un modello ONNX leggero (~26 MB, zero PyTorch, zero token) e, a fine trascrizione, consente di cliccare **Rinomina voci** per sostituire i nomi sia sullo schermo sia nel file salvato, anche dopo la correzione PRO.
- **Parakeet TDT 0.6B v3**: modello selezionabile dal menu (ONNX via `onnx-asr`, ~25 lingue europee con rilevamento automatico, ~2,5 GB scaricati al primo uso). Gira su CPU (con `onnxruntime-gpu` può usare la GPU), non usa il contesto/prompt personalizzato né la scelta lingua, e lavora a blocchi di 25 s. Supporta riconoscimento voci, timestamp ed export `.txt`.
- **LIVE (Microfono)**: pulsante accanto a *Sfoglia file...* per trascrivere in tempo reale dal microfono (solo Linux: richiede `pw-record`, `parec` o `arecord`). Premendo **Stop LIVE** viene salvato `Trascrizioni/LIVE_<data>.txt`. Usa NVIDIA Nemotron 3.5 Streaming (cache-aware, aggiornamento ~1,3 s, caricamento iniziale ~1–2 min) se nel `.venv` sono installati `torch`, `torchaudio`, `nemo_toolkit[asr]` e `matplotlib`; altrimenti ripiega su faster-whisper `large-v3-turbo` (blocchi ~3 s). Il LIVE non parte se è in corso una trascrizione di file.
- **Visualizzazione live & export automatico**:
  - Streaming del testo in tempo reale raggruppato in paragrafi leggibili.
  - Toggle **Mostra timestamp** `(MM:SS)` senza ripetere la trascrizione.
  - Salvataggio automatico di `<nome_file>.txt` e, con PRO attivo, `<nome_file>.pro.txt` nella cartella `Trascrizioni/`.
  - Pulsanti rapidi *Copia testo*, *Apri .txt*, *Rinomina voci*, *Apri cartella Trascrizioni* e *Annulla*.
- **Gestione pulita della GPU**: rilascio automatico della VRAM al termine della coda o in caso di annullamento/chiusura.
- **Istanza singola e scorciatoie**: riaprire l'app porta in primo piano la finestra esistente; chiusura rapida con `Ctrl+W` o `Ctrl+Q` (con conferma se una trascrizione è in corso).
- **Script CLI alternativo (`trascrivi.py`)**: per trascrivere rapidamente da terminale o tramite selettore `zenity`.

---

## Chi può usarlo e compatibilità dispositivi

### A chi è pensato
- **Studenti universitari** (in particolare Medicina, Chirurgia, Professioni Sanitarie e facoltà tecnico-scientifiche) per sbobinare lezioni lunghe senza limiti di minuti né abbonamenti cloud.
- **Ricercatori, giornalisti, professionisti e docenti** che trattano interviste, riunioni o appunti vocali riservati e vogliono che i file **non lascino mai il proprio computer**.

### Sistemi operativi supportati
| Sistema Operativo | App Desktop (`app.py`) | CLI (`trascrivi.py`) | Note |
| --- | :---: | :---: | --- |
| **Linux** (Ubuntu 22.04/24.04+, Fedora, Debian, Arch con GNOME/Wayland o X11) | ✅ Pieno supporto | ✅ Pieno supporto | Consigliato GTK 4.12+ e Libadwaita 1.4+ |
| **Windows 11** (**WSL2 + WSLg** per l'app completa; Python nativo per la CLI) | ✅ Tramite WSL2 | ✅ | WSLg mostra la GUI su Windows; CUDA richiede driver NVIDIA compatibili. Il percorso MSYS2 è avanzato e non garantisce tutte le dipendenze |
| **macOS** (Apple Silicon M1/M2/M3/M4 e Intel tramite **Homebrew**) | ✅ Supportata | ✅ Pieno supporto | GTK 4 + Libadwaita via Homebrew; trascrizione in modalità CPU `int8` automatica |

### Requisiti hardware (GPU / CPU / RAM)
| Hardware | Esperienza d'uso | Modello consigliato | Requisiti indicativi |
| --- | --- | --- | --- |
| **GPU NVIDIA con CUDA 12+** *(consigliato)* | Molto veloce (molte volte più rapido del tempo reale, `float16`) | `large-v3`, `Ensemble` o `large-v3-turbo` | Driver NVIDIA aggiornati; **≥ 6–8 GB VRAM** per `large-v3` / `Ensemble`, **≥ 4 GB VRAM** per `large-v3-turbo` |
| **Solo CPU (senza GPU NVIDIA)** | Fallback automatico `int8` (più lento ma pienamente funzionante) | `large-v3-turbo` | CPU multi-core moderna (o Apple Silicon Serie M), **≥ 8–16 GB RAM** |
| **GPU AMD / Intel / Apple Silicon** | Usa automaticamente il fallback CPU `int8` | `large-v3-turbo` | CTranslate2 utilizza CUDA su NVIDIA |

> **Spazio su disco**: alla prima esecuzione i modelli Whisper vengono scaricati automaticamente da Hugging Face nella cache locale (~1,6 GB per `large-v3-turbo`, ~3 GB per `large-v3`). Le trascrizioni successive funzionano completamente offline.

---

## Formati audio/video supportati

Supporta tutti i principali formati decodificabili da PyAV/FFmpeg:
`.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.ogg`, `.flac`, `.webm`, `.aac`, `.opus`, `.mov`, `.avi`, `.wma`, `.mpeg`, `.mpg`.

---

## Installazione modelli locali PRO

PRO usa un modello locale. L'audio resta sul PC: Whisper lo trascrive e il testo
viene inviato solo al server locale del provider scelto.

### Ollama

1. Installa [Ollama](https://ollama.com/download).
2. Scarica un modello di chat, per esempio:

   ```bash
   ollama pull qwen3:8b
   ```

Quando attivi PRO, l'app avvia `ollama serve` in background se il servizio non è
già attivo, elenca i modelli installati e carica quello scelto.

### LM Studio

1. Installa [LM Studio](https://lmstudio.ai/) e scarica un modello di chat.
2. Abilita il comando CLI `lms` nelle impostazioni di LM Studio. Per il
   funzionamento senza GUI, consulta la [guida headless di LM Studio](https://lmstudio.ai/docs/developer/core/headless).
3. Attiva **PRO**: l'app avvia il daemon headless `lms`, elenca i modelli
   installati e carica quello scelto. Non apre la GUI e non scarica modelli.
   Il caricamento può richiedere alcuni minuti.

Se è installato un solo modello chat, viene scelto automaticamente; con più
modelli compare un selettore. Gli embedding non sono proposti. Un modello Qwen
7B *Instruct* è una buona scelta per la correzione; un modello *Coder* può
funzionare, ma un modello generalista è spesso più adatto alle trascrizioni.
L'inventario automatico supporta LM Studio e Ollama; altri runtime richiedono un
adapter specifico, perché non esiste un catalogo locale universale.

Se nessun modello è installato, l'app mostra la guida. Per disattivare la
correzione, spegni l'interruttore PRO; la trascrizione standard non cambia.

---

## Installazione su Linux (Ubuntu / Debian)

### 1. Dipendenze di sistema (GTK 4 + Libadwaita)
Su Ubuntu 24.04 GNOME sono spesso già presenti; in caso contrario installale con:

```bash
sudo apt update
sudo apt install python3-venv python3-gi gir1.2-gtk-4.0 gir1.2-adw-1
```

*(Su altre distribuzioni installa i corrispondenti pacchetti `pygobject`, `gtk4` e `libadwaita`.)*

### 2. Clona il repository e crea l'ambiente virtuale

```bash
git clone https://github.com/GandalfSilente/audio-transcription-PRO.git
cd audio-transcription-PRO
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

> **Solo CPU (senza GPU NVIDIA)**: installa `.venv/bin/pip install faster-whisper onnx-asr` invece di `requirements.txt`; così mantieni Whisper e Parakeet senza installare i pacchetti CUDA NVIDIA.
>
> **Opzionale — Nemotron 3.5 LIVE nativo (GPU CUDA)**: per usare `nvidia/nemotron-3.5-asr-streaming-0.6b` al posto del fallback Whisper nel tasto **LIVE (Microfono)**:
> ```bash
> .venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
> .venv/bin/pip install "nemo_toolkit[asr]" matplotlib
> ```

---

## Installazione su Windows 11

Su Windows 11 puoi usare l'applicazione in due modi (senza modificare il codice):

### Metodo A — Consigliato (App Desktop GUI + GPU NVIDIA CUDA): tramite WSL2 + WSLg
Windows 11 include nativamente **WSLg** (che mostra le finestre grafiche Linux direttamente sul desktop di Windows) e supporta il passthrough automatico dei driver **NVIDIA CUDA** installati su Windows.

1. Assicurati di avere i **driver NVIDIA per Windows** aggiornati.
2. Apri **PowerShell** e installa WSL2 con Ubuntu (se non è già presente):
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. Apri il terminale **Ubuntu** da Windows ed esegui esattamente i comandi della guida Linux:
   ```bash
   sudo apt update
   sudo apt install git python3-venv python3-gi gir1.2-gtk-4.0 gir1.2-adw-1
   git clone https://github.com/GandalfSilente/audio-transcription-PRO.git
   cd audio-transcription-PRO
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install -r requirements.txt
   .venv/bin/python app.py
   ```
   *(La finestra dell'app apparirà direttamente sul desktop di Windows 11; i file del tuo PC Windows sono accessibili in `/mnt/c/Users/<TuoUtente>/Downloads`.)*
4. Per avviare l'app con doppio clic, usa il file [`avvia-pro-wsl.bat`](avvia-pro-wsl.bat) dal repository in Windows. Di default avvia `~/audio-transcription-PRO` nella distro `Ubuntu-24.04`; modifica le prime due variabili del `.bat` se hai scelto nomi o percorsi diversi. WSL e l'ambiente `.venv` devono essere già installati con i passaggi sopra.

Per usare PRO in WSL2, installa Ollama o LM Studio **nella stessa distro WSL** e scarica lì il modello: i servizi installati solo su Windows non sono automaticamente raggiungibili dall'app Linux tramite `127.0.0.1`.

### Metodo B — Windows 11 nativo (senza WSL)
- **Solo CLI (`trascrivi.py`) con Python per Windows**:
  ```powershell
  git clone https://github.com/GandalfSilente/audio-transcription-PRO.git
  cd audio-transcription-PRO
  python -m venv .venv
  .venv\Scripts\pip install --upgrade pip
  .venv\Scripts\pip install faster-whisper
  .venv\Scripts\python trascrivi.py "C:\percorso\del\file.mp3"
  ```
  *(Questa modalità CLI non include l'interfaccia GTK né la correzione PRO. Per l'app completa usa WSL2 + WSLg. Per usare CUDA su Windows nativo installa CUDA 12 + cuBLAS/cuDNN per Windows oppure aggiungi le DLL NVIDIA al `PATH`; altrimenti userà automaticamente la CPU in `int8`.)*
- **App Desktop GUI (`app.py`) tramite MSYS2 (UCRT64)**:

  *(Percorso nativo avanzato: l'app e le dipendenze devono essere compatibili con i pacchetti MSYS2 installati; per un'installazione completa e riproducibile, incluso PRO, è preferibile WSL2 + WSLg.)*
  1. Installa [MSYS2](https://www.msys2.org/) e apri il terminale **MSYS2 UCRT64**.
  2. Installa Python, GTK 4, Libadwaita e PyGObject:
     ```bash
     pacman -S mingw-w64-ucrt-x86_64-python mingw-w64-ucrt-x86_64-python-pip \
               mingw-w64-ucrt-x86_64-python-gobject mingw-w64-ucrt-x86_64-gtk4 \
               mingw-w64-ucrt-x86_64-libadwaita mingw-w64-ucrt-x86_64-adwaita-icon-theme
     ```
  3. Clona il progetto e crea la `.venv` condividendo i pacchetti GTK di sistema:
     ```bash
     git clone https://github.com/GandalfSilente/audio-transcription-PRO.git
     cd audio-transcription-PRO
     python3 -m venv --system-site-packages .venv
     .venv/bin/pip install faster-whisper onnx-asr
     .venv/bin/python app.py
     ```

---

## Installazione su macOS (Apple Silicon M1/M2/M3/M4 e Intel)

Su macOS l'interfaccia GTK 4 + Libadwaita funziona nativamente tramite **Homebrew** e `faster-whisper` utilizza automaticamente la CPU in modalità `int8`.

1. Installa [Homebrew](https://brew.sh/) (se non è già installato).
2. Installa Python, GTK 4, Libadwaita e PyGObject:
   ```bash
   brew update
   brew install python@3.12 pygobject3 gtk4 libadwaita adwaita-icon-theme
   ```
3. Clona il repository e crea la `.venv` con `--system-site-packages` (così Python vede `gi` / GTK 4 installati da Homebrew):
   ```bash
   git clone https://github.com/GandalfSilente/audio-transcription-PRO.git
   cd audio-transcription-PRO
   $(brew --prefix python@3.12)/bin/python3.12 -m venv --system-site-packages .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install faster-whisper onnx-asr
   ```
4. Avvia l'app desktop (o lo script CLI):
   ```bash
   .venv/bin/python app.py
   # oppure da terminale:
   .venv/bin/python trascrivi.py /percorso/del/file.m4a
   ```
   *(Installa `faster-whisper` e `onnx-asr` per avere Whisper e Parakeet. Su Mac consigliamo `large-v3-turbo` per prestazioni migliori su CPU. Per PRO installa Ollama o LM Studio e un modello locale.)*

---

## Utilizzo

### Avvio dell'app desktop
```bash
.venv/bin/python app.py
```

Puoi anche passare uno o più file direttamente da riga di comando:
```bash
.venv/bin/python app.py lezione1.mp3 lezione2.mp4
```

### Avvio della versione CLI
```bash
.venv/bin/python trascrivi.py percorso/del/file.mp3
```
Se eseguito senza argomenti, `trascrivi.py` apre il selettore file (`zenity`, se installato) oppure chiede di trascinare il percorso nel terminale e salva `.txt` e `.srt` accanto al file sorgente.

---

## Creare l'icona sul Desktop (avvio con doppio clic senza terminale)

### Linux (Ubuntu / GNOME / KDE)
Crea un file `~/Desktop/Trascrizione-Audio.desktop` (e/o in `~/.local/share/applications/Trascrizione-Audio.desktop` per averla anche nel menu applicazioni) sostituendo `/PERCORSO/DEL/PROGETTO` con la cartella in cui hai clonato il repository:

```ini
[Desktop Entry]
Type=Application
Name=Trascrizione Audio
Comment=Trascrivi audio e video localmente (App Desktop Nativa)
Exec=/PERCORSO/DEL/PROGETTO/.venv/bin/python /PERCORSO/DEL/PROGETTO/app.py %F
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Utility;
StartupWMClass=org.trascrizioneaudio.Pro
```

Rendi eseguibile il file e (su Ubuntu/GNOME) abilita l'avvio da Desktop:
```bash
chmod +x ~/Desktop/Trascrizione-Audio.desktop
gio set ~/Desktop/Trascrizione-Audio.desktop metadata::trusted true
```
*(In alternativa su Ubuntu fai tasto destro sull'icona sul Desktop → **Consenti avvio** / **Allow Launching**.)*

### Windows 11
- **Se usi il Metodo A (WSL2 + WSLg — con GPU NVIDIA CUDA)**:
  1. Fai **tasto destro sul Desktop** di Windows 11 → **Nuovo → Collegamento**.
  2. Nel campo *Immettere il percorso per il collegamento* incolla (sostituendo `~/audio-transcription-PRO` se hai clonato il progetto in un altro percorso dentro WSL):
     ```text
     wslg.exe -d Ubuntu-24.04 --cd "~/audio-transcription-PRO" -- ./.venv/bin/python app.py
     ```
     *(Usando `wslg.exe` invece di `wsl.exe` l'app si apre direttamente come finestra grafica su Windows 11 **senza aprire alcuna finestra del terminale**).*
  3. Clicca **Avanti**, assegna il nome **Trascrizione Audio** e clicca **Fine**.
  4. *(Opzionale)* Tasto destro sul collegamento → **Proprietà → Cambia icona...** per scegliere un'icona personalizzata (es. da `%SystemRoot%\System32\shell32.dll` o `imageres.dll`).
- **Se usi il Metodo B (MSYS2 UCRT64 nativo)**:
  1. Fai **tasto destro sul Desktop** → **Nuovo → Collegamento**.
  2. Come percorso inserisci (adattando `C:\percorso\audio-transcription-PRO`):
     ```text
     C:\msys64\ucrt64.exe C:\percorso\audio-transcription-PRO\.venv\bin\pythonw.exe C:\percorso\audio-transcription-PRO\app.py
     ```
     *(L'eseguibile `pythonw.exe` avvia la finestra grafica senza mostrare il prompt dei comandi).*

### macOS (Icona `.app` nativa sul Desktop e nel Dock)
Su macOS puoi generare un'applicazione nativa **`Trascrizione Audio.app`** direttamente sul Desktop (senza aprire il Terminale all'avvio) con un singolo comando usando `osacompile` (incluso di serie in macOS).

Apri il Terminale ed esegui (adattando `~/audio-transcription-PRO` se hai clonato il repository altrove):

```bash
osacompile -o ~/Desktop/"Trascrizione Audio.app" \
  -e 'do shell script "export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH; cd ~/audio-transcription-PRO && ./.venv/bin/python app.py >/dev/null 2>&1 &"'
```

- Sul **Desktop** comparirà subito l'app **Trascrizione Audio**: basta un **doppio clic** per aprire direttamente la finestra dell'applicazione senza alcuna finestra del Terminale.
- Puoi anche trascinare `Trascrizione Audio.app` nella cartella **Applicazioni** (`/Applications`) o direttamente nella barra **Dock** di macOS.

---

## Struttura del progetto

- `app.py` — Interfaccia desktop nativa GTK 4 + Libadwaita (Drag & Drop, coda, profili di contesto personalizzabili, interruttore PRO, visualizzazione live, scorciatoie `Ctrl+W`/`Ctrl+Q`).
- `engine.py` — Motore di trascrizione disaccoppiato dalla GUI (lazy loading CUDA/`faster-whisper`, gestione coda FIFO, preset/persistenza XDG, VAD d'aula, modalità Ensemble, annullamento e rilascio VRAM).
- `pro.py` — Rilevamento dei modelli Ollama/LM Studio e correzione locale delle trascrizioni.
- `diarizer.py` — Motore di riconoscimento voci ONNX leggero (`WeSpeaker ResNet34` via `onnxruntime` + clustering NumPy, senza PyTorch).
- `live.py` — Trascrizione LIVE da microfono (cattura audio a chunk, Nemotron o fallback Whisper).
- `trascrivi.py` — Utility CLI indipendente con `large-v3-turbo`.
- `avvia-pro-wsl.bat` — Launcher Windows per aprire l'app completa nell'ambiente WSL2 configurato.
- `requirements.txt` — Dipendenze Python per la `.venv`.
- `Trascrizioni/` — Cartella generata automaticamente per i file `.txt` prodotti (ignorata da Git).

---

## Privacy

- L'app elabora localmente audio, video e trascrizioni; non li invia a servizi cloud.
- Il primo utilizzo o l'installazione di modelli aggiuntivi può richiedere Internet per scaricare pesi e dipendenze (per esempio da Hugging Face). Questi download non includono le registrazioni o le trascrizioni. Dopo aver scaricato i modelli necessari, l'inferenza può funzionare offline.
- Con PRO attivo, la trascrizione e il contesto selezionato vengono inviati al modello tramite l'API locale di Ollama (`127.0.0.1:11434`) o LM Studio (`127.0.0.1:1234`). L'app non invia questo testo a un servizio LLM cloud; verifica comunque le impostazioni e l'informativa del runtime locale installato.
- I file originali non vengono modificati o eliminati. Le trascrizioni generate sono salvate localmente in `Trascrizioni/`.
- Profili e preferenze sono salvati in `~/.config/trascrizione-audio/config.json`, fuori dal repository.
- `Trascrizioni/` è esclusa da Git tramite `.gitignore`: questo evita il tracciamento Git, ma non cifra né cancella i file locali.

---

## Licenza e Disclaimer

- **Doppia licenza**: il codice sorgente è disponibile, a scelta dell'utente, secondo i termini di [MIT](LICENSE-MIT) oppure [Apache License 2.0](LICENSE-APACHE).
- **Componenti di terze parti**: i modelli e le librerie esterni eventualmente scaricati o utilizzati (tra cui Whisper / `faster-whisper`, Parakeet, WeSpeaker e Nemotron) restano soggetti alle rispettive licenze originali.
- **Esclusione di responsabilità**: il software è fornito *"così com'è"* (*"AS IS"*), senza alcuna garanzia esplicita o implicita. Gli autori non assumono alcuna responsabilità per eventuali danni o usi impropri: l'utente finale è l'unico responsabile dell'utilizzo dello strumento e del pieno rispetto delle normative vigenti (incluse quelle su privacy, consenso alle registrazioni audio/video e diritto d'autore).
