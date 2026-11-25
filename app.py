#!/usr/bin/env python3
from __future__ import annotations

import sys
import threading
from pathlib import Path

try:
    import gi
except ImportError:
    sys.path.append("/usr/lib/python3/dist-packages")
    import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gtk, Pango

from pipeline import (
    ASR_MODEL,
    OLLAMA_MODEL,
    correct_transcript,
    next_corrected_path,
    transcribe_audio,
    transcript_paths,
)

PROJECT_DIR = Path(__file__).resolve().parent
APP_CSS = """
window.background {
    background-color: #0f1117;
    color: #e6eaf2;
}

headerbar {
    background-color: #141821;
    border-bottom: 1px solid #232a3a;
    box-shadow: none;
}

.card-panel {
    background-color: #181c26;
    border: 1px solid #2a3142;
    border-radius: 12px;
    padding: 14px 16px;
}

.muted-label {
    color: #8b95a8;
    font-size: 13px;
}

.field-label {
    color: #c5cedf;
    font-size: 13px;
    font-weight: 600;
}

.transcript-scroll {
    background-color: #181c26;
    border: 1px solid #2a3142;
    border-radius: 12px;
}

.transcript-view {
    background-color: #181c26;
    color: #e6eaf2;
    font-size: 15px;
}

.transcript-view text {
    background-color: #181c26;
    color: #e6eaf2;
}

button.accent-btn {
    background-color: #4f8cff;
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
}

button.accent-btn:hover {
    background-color: #649bff;
}

button.secondary-btn {
    background-color: #181c26;
    color: #e6eaf2;
    border: 1px solid #2a3142;
    border-radius: 8px;
}

button.secondary-btn:hover {
    background-color: #202738;
    border-color: #3a4663;
}

button.pro-btn {
    background-color: #263d67;
    color: #a9c7ff;
    border: 1px solid #4f8cff;
    border-radius: 8px;
    font-weight: 700;
}

button.pro-btn:hover {
    background-color: #304d7d;
}
"""


class TranscriptionApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id="org.trascrizioneaudio.Pro")
        self.audio_path: Path | None = None
        self.current_audio: Path | None = None
        self.current_transcript: str | None = None
        self.current_raw_path: Path | None = None
        self.current_corrected_path: Path | None = None
        self._pulse_source_id: int | None = None

    def do_activate(self) -> None:
        Adw.StyleManager.get_default().set_color_scheme(
            Adw.ColorScheme.FORCE_DARK
        )
        display = Gdk.Display.get_default()
        if display:
            provider = Gtk.CssProvider()
            provider.load_from_data(APP_CSS.encode())
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

        self.window = Adw.ApplicationWindow(application=self)
        self.window.set_title("Trascrizione Audio Locale")
        self.window.set_default_size(1100, 800)

        overlay = Adw.ToastOverlay()
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        overlay.set_child(root)
        self.window.set_content(overlay)

        header = Adw.HeaderBar()
        header.set_title_widget(
            Adw.WindowTitle(
                title="Trascrizione Audio Locale",
                subtitle="GPU CUDA / CPU • Whisper + correzione LLM locale",
            )
        )
        root.append(header)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(20)
        content.set_margin_end(20)
        content.set_vexpand(True)
        root.append(content)

        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        controls.add_css_class("card-panel")
        content.append(controls)

        file_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        controls.append(file_row)
        self.file_label = Gtk.Label(label="Nessun file selezionato")
        self.file_label.add_css_class("muted-label")
        self.file_label.set_xalign(0)
        self.file_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self.file_label.set_hexpand(True)
        file_row.append(self.file_label)
        self.file_button = Gtk.Button(label="Sfoglia file…")
        self.file_button.add_css_class("secondary-btn")
        self.file_button.connect("clicked", self._choose_file)
        file_row.append(self.file_button)

        settings = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        controls.append(settings)
        asr_label = Gtk.Label(label=f"Whisper: {ASR_MODEL} • Italiano")
        asr_label.add_css_class("field-label")
        settings.append(asr_label)
        settings.append(Gtk.Label(label="LLM locale:"))
        self.model_entry = Gtk.Entry()
        self.model_entry.set_text(OLLAMA_MODEL)
        self.model_entry.set_hexpand(True)
        self.model_entry.set_tooltip_text(
            "Nome di un modello già installato in Ollama"
        )
        settings.append(self.model_entry)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        controls.append(actions)
        self.transcribe_button = Gtk.Button(label="Trascrivi")
        self.transcribe_button.add_css_class("accent-btn")
        self.transcribe_button.connect("clicked", lambda _button: self._start(False))
        actions.append(self.transcribe_button)
        self.pro_button = Gtk.Button(label="PRO")
        self.pro_button.add_css_class("pro-btn")
        self.pro_button.set_tooltip_text(
            "Correggi gli errori e migliora la leggibilità con un LLM locale"
        )
        self.pro_button.connect("clicked", lambda _button: self._start(True))
        actions.append(self.pro_button)
        pro_hint = Gtk.Label(
            label="PRO trascrive se necessario, poi migliora il testo in locale."
        )
        pro_hint.add_css_class("muted-label")
        pro_hint.set_hexpand(True)
        pro_hint.set_xalign(0)
        actions.append(pro_hint)

        self.status = Gtk.Label(label="Pronto")
        self.status.add_css_class("muted-label")
        self.status.set_xalign(0)
        self.status.set_wrap(True)
        content.append(self.status)
        self.progress = Gtk.ProgressBar()
        content.append(self.progress)

        results = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        results.set_vexpand(True)
        content.append(results)
        raw_panel, self.raw_buffer = self._transcript_panel("Trascrizione originale")
        corrected_panel, self.corrected_buffer = self._transcript_panel(
            "Testo corretto • PRO"
        )
        results.set_start_child(raw_panel)
        results.set_end_child(corrected_panel)
        results.set_position(530)

        self.window.present()

    @staticmethod
    def _transcript_panel(title: str) -> tuple[Gtk.Widget, Gtk.TextBuffer]:
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        panel.set_margin_start(4)
        panel.set_margin_end(4)
        heading = Gtk.Label(label=title)
        heading.add_css_class("field-label")
        heading.set_xalign(0)
        panel.append(heading)

        scroll = Gtk.ScrolledWindow()
        scroll.add_css_class("transcript-scroll")
        scroll.set_vexpand(True)
        view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.add_css_class("transcript-view")
        view.set_editable(False)
        view.set_cursor_visible(False)
        view.set_top_margin(10)
        view.set_bottom_margin(10)
        view.set_left_margin(12)
        view.set_right_margin(12)
        scroll.set_child(view)
        panel.append(scroll)
        return panel, view.get_buffer()

    def _choose_file(self, _button: Gtk.Button) -> None:
        chooser = Gtk.FileChooserNative.new(
            "Scegli file audio o video",
            self.window,
            Gtk.FileChooserAction.OPEN,
            "Apri",
            "Annulla",
        )
        chooser.connect("response", self._on_file_chosen)
        chooser.show()

    def _on_file_chosen(self, chooser: Gtk.FileChooserNative, response: int) -> None:
        if response == Gtk.ResponseType.ACCEPT:
            file = chooser.get_file()
            if file and file.get_path():
                selected = Path(file.get_path())
                self.audio_path = selected
                self.file_label.set_text(str(selected))
                if selected != self.current_audio:
                    self.current_audio = None
                    self.current_transcript = None
                    self.current_raw_path = None
                    self.current_corrected_path = None
                    self.raw_buffer.set_text("")
                    self.corrected_buffer.set_text("")
        chooser.destroy()

    def _start(self, use_pro: bool) -> None:
        audio_path = self.audio_path
        if audio_path is None:
            self.status.set_text("Seleziona prima un file audio.")
            return

        model = self.model_entry.get_text().strip()
        if use_pro and not model:
            self.status.set_text("Inserisci il nome di un modello Ollama installato.")
            return

        reuse_transcript = (
            use_pro
            and self.current_audio == audio_path
            and self.current_transcript is not None
        )
        transcript = self.current_transcript if reuse_transcript else None
        raw_path = self.current_raw_path if reuse_transcript else None
        corrected_path = (
            self.current_corrected_path if reuse_transcript else None
        )
        if reuse_transcript and raw_path and corrected_path and corrected_path.exists():
            corrected_path = next_corrected_path(raw_path)

        self.raw_buffer.set_text("" if not reuse_transcript else self.current_transcript)
        self.corrected_buffer.set_text("")
        self._set_busy(True)
        self.status.set_text(
            "Correzione PRO locale in corso…"
            if transcript is not None
            else "Trascrizione audio in corso…"
        )
        threading.Thread(
            target=self._run_pipeline,
            args=(audio_path, use_pro, model, transcript, raw_path, corrected_path),
            daemon=True,
        ).start()

    def _run_pipeline(
        self,
        audio_path: Path,
        use_pro: bool,
        model: str,
        transcript: str | None,
        raw_path: Path | None,
        corrected_path: Path | None,
    ) -> None:
        try:
            if transcript is None:
                transcript = transcribe_audio(audio_path)
                raw_output, corrected_path = transcript_paths(
                    PROJECT_DIR, audio_path
                )
                raw_output.write_text(transcript + "\n", encoding="utf-8")
                raw_path = raw_output
                GLib.idle_add(
                    self._show_raw, transcript, audio_path, raw_path, corrected_path
                )

            if not use_pro:
                GLib.idle_add(
                    self._finish,
                    None,
                    f"Trascrizione salvata: {raw_path.name}",
                    None,
                )
                return

            self._set_status_from_worker("Correzione e miglioramento leggibilità…")
            corrected = correct_transcript(
                transcript,
                model,
                progress=lambda message: GLib.idle_add(self.status.set_text, message),
            )
            if raw_path is None or corrected_path is None:
                raise RuntimeError("Percorso di salvataggio della trascrizione mancante.")
            corrected_path.write_text(corrected + "\n", encoding="utf-8")
            GLib.idle_add(
                self._finish,
                corrected,
                f"Completato. Originale: {raw_path.name}; PRO: {corrected_path.name}",
                corrected_path,
            )
        except Exception as error:
            partial = f" Originale salvato: {raw_path.name}." if raw_path else ""
            GLib.idle_add(self._finish, None, f"Errore: {error}.{partial}", None)

    def _show_raw(
        self,
        text: str,
        audio_path: Path,
        raw_path: Path,
        corrected_path: Path,
    ) -> None:
        self.current_audio = audio_path
        self.current_transcript = text
        self.current_raw_path = raw_path
        self.current_corrected_path = corrected_path
        self.raw_buffer.set_text(text)

    def _set_status_from_worker(self, message: str) -> None:
        GLib.idle_add(self.status.set_text, message)

    def _set_busy(self, busy: bool) -> None:
        self.file_button.set_sensitive(not busy)
        self.transcribe_button.set_sensitive(not busy)
        self.pro_button.set_sensitive(not busy)
        self.model_entry.set_sensitive(not busy)
        if busy:
            self._pulse_source_id = GLib.timeout_add(100, self._pulse_progress)
        elif self._pulse_source_id is not None:
            GLib.source_remove(self._pulse_source_id)
            self._pulse_source_id = None
            self.progress.set_fraction(0)

    def _pulse_progress(self) -> bool:
        self.progress.pulse()
        return True

    def _finish(
        self, corrected: str | None, message: str, corrected_path: Path | None
    ) -> None:
        if corrected is not None:
            self.corrected_buffer.set_text(corrected)
            self.current_corrected_path = corrected_path
        self.status.set_text(message)
        self._set_busy(False)


if __name__ == "__main__":
    TranscriptionApp().run()
