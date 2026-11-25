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
from gi.repository import GLib, Gtk, Pango

from pipeline import (
    ASR_MODEL,
    OLLAMA_MODEL,
    correct_transcript,
    transcribe_audio,
    transcript_paths,
)

PROJECT_DIR = Path(__file__).resolve().parent


class TranscriptionApp(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="org.trascrizioneaudio.Pro")
        self.audio_path: Path | None = None

    def do_activate(self) -> None:
        window = Gtk.ApplicationWindow(application=self)
        window.set_title("Trascrizioni Audio PRO")
        window.set_default_size(1100, 760)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        root.set_margin_top(20)
        root.set_margin_bottom(20)
        root.set_margin_start(20)
        root.set_margin_end(20)
        window.set_child(root)

        heading = Gtk.Label(label="Trascrizione locale in due passaggi")
        heading.add_css_class("title-1")
        heading.set_xalign(0)
        root.append(heading)

        description = Gtk.Label(
            label=(
                "Whisper trascrive l'audio; Ollama corregge il testo in locale. "
                "L'originale resta intatto."
            )
        )
        description.set_xalign(0)
        description.set_wrap(True)
        root.append(description)

        file_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        root.append(file_row)
        self.file_label = Gtk.Label(label="Nessun file selezionato")
        self.file_label.set_xalign(0)
        self.file_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.file_label.set_hexpand(True)
        file_row.append(self.file_label)
        choose_button = Gtk.Button(label="Scegli audio…")
        choose_button.connect("clicked", self._choose_file, window)
        file_row.append(choose_button)

        settings = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        root.append(settings)
        settings.append(Gtk.Label(label=f"Whisper: {ASR_MODEL}"))
        settings.append(Gtk.Label(label="Lingua: Italiano"))
        settings.append(Gtk.Label(label="Ollama:"))
        self.model_entry = Gtk.Entry()
        self.model_entry.set_text(OLLAMA_MODEL)
        self.model_entry.set_hexpand(True)
        self.model_entry.set_tooltip_text("Nome del modello locale installato in Ollama")
        settings.append(self.model_entry)

        self.start_button = Gtk.Button(label="Trascrivi e correggi")
        self.start_button.add_css_class("suggested-action")
        self.start_button.connect("clicked", self._start)
        root.append(self.start_button)

        self.status = Gtk.Label(label="Pronto")
        self.status.set_xalign(0)
        self.status.set_wrap(True)
        root.append(self.status)

        results = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        results.set_vexpand(True)
        root.append(results)
        raw_panel, self.raw_buffer = self._transcript_panel(
            "Trascrizione originale"
        )
        corrected_panel, self.corrected_buffer = self._transcript_panel(
            "Testo corretto"
        )
        results.set_start_child(raw_panel)
        results.set_end_child(corrected_panel)
        results.set_position(540)

        window.present()

    @staticmethod
    def _transcript_panel(title: str) -> tuple[Gtk.Widget, Gtk.TextBuffer]:
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        label = Gtk.Label(label=title)
        label.set_xalign(0)
        label.add_css_class("heading")
        panel.append(label)
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.set_editable(False)
        view.set_cursor_visible(False)
        scroll.set_child(view)
        panel.append(scroll)
        return panel, view.get_buffer()

    def _choose_file(self, _button: Gtk.Button, parent: Gtk.Window) -> None:
        chooser = Gtk.FileChooserNative.new(
            "Scegli file audio o video",
            parent,
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
                self.audio_path = Path(file.get_path())
                self.file_label.set_text(str(self.audio_path))
        chooser.destroy()

    def _start(self, _button: Gtk.Button) -> None:
        if self.audio_path is None:
            self.status.set_text("Seleziona prima un file audio.")
            return
        model = self.model_entry.get_text().strip()
        if not model:
            self.status.set_text("Inserisci il nome di un modello Ollama.")
            return

        self.raw_buffer.set_text("")
        self.corrected_buffer.set_text("")
        self.start_button.set_sensitive(False)
        self.status.set_text("Trascrizione audio in corso…")
        threading.Thread(
            target=self._run_pipeline,
            args=(self.audio_path, model),
            daemon=True,
        ).start()

    def _run_pipeline(self, audio_path: Path, model: str) -> None:
        raw_path: Path | None = None
        try:
            raw = transcribe_audio(audio_path)
            raw_output, corrected_path = transcript_paths(PROJECT_DIR, audio_path)
            raw_output.write_text(raw + "\n", encoding="utf-8")
            raw_path = raw_output
            GLib.idle_add(self._show_raw, raw, raw_path)
            corrected = correct_transcript(
                raw, model, progress=lambda message: GLib.idle_add(
                    self.status.set_text, message
                )
            )
            corrected_path.write_text(corrected + "\n", encoding="utf-8")
            GLib.idle_add(
                self._finish,
                corrected,
                f"Completato. Originale: {raw_path.name}; corretto: {corrected_path.name}",
            )
        except Exception as error:
            partial = (
                f" Trascrizione originale salvata in {raw_path.name}."
                if raw_path
                else ""
            )
            GLib.idle_add(self._finish, None, f"Errore: {error}.{partial}")

    def _show_raw(self, text: str, path: Path) -> None:
        self.raw_buffer.set_text(text)
        self.status.set_text(
            f"Trascrizione salvata ({path.name}). Correzione locale in corso…"
        )

    def _finish(self, corrected: str | None, message: str) -> None:
        if corrected is not None:
            self.corrected_buffer.set_text(corrected)
        self.status.set_text(message)
        self.start_button.set_sensitive(True)


if __name__ == "__main__":
    TranscriptionApp().run()
