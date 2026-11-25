#!/usr/bin/env python3
"""Native Linux desktop application (GTK 4 + Libadwaita) for faster-whisper transcription."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

# Allow .venv Python to access Ubuntu's system PyGObject (GTK 4 / Libadwaita)
# while preserving .venv package priority (faster-whisper, ctranslate2, CUDA).
try:
    import gi
except ImportError:
    dist_packages = "/usr/lib/python3/dist-packages"
    if dist_packages not in sys.path:
        sys.path.append(dist_packages)
    import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

import engine  # noqa: E402

APP_ID = "org.trascrizioneaudio.Pro"
APP_TITLE = "Trascrizione Audio PRO"

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

.hint-label {
    color: #8b95a8;
    font-size: 12px;
}

.field-label {
    color: #c5cedf;
    font-size: 13px;
    font-weight: 600;
}

.prompt-scroll {
    background-color: #111520;
    border: 1px solid #2a3142;
    border-radius: 8px;
}

.prompt-view {
    background-color: #111520;
    color: #e6eaf2;
    font-size: 13px;
}

.prompt-view text {
    background-color: #111520;
    color: #e6eaf2;
}

.drop-zone {
    background-color: #181c26;
    border: 2px dashed #2e3850;
    border-radius: 14px;
    padding: 24px 20px;
    transition: background-color 140ms ease, border-color 140ms ease;
}

.drop-zone:hover {
    border-color: #4264a8;
    background-color: #1b2232;
}

.drop-zone.drop-over {
    border-color: #4f8cff;
    background-color: #1e2942;
}

.drop-title {
    font-size: 16px;
    font-weight: 700;
    color: #e6eaf2;
}

.drop-subtitle {
    font-size: 13px;
    color: #8b95a8;
}

progressbar trough {
    min-height: 10px;
    border-radius: 999px;
    background-color: #181c26;
    border: 1px solid #2a3142;
}

progressbar progress {
    min-height: 10px;
    border-radius: 999px;
    background-color: #4f8cff;
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

.transcript-view.placeholder-text {
    color: #7c879c;
}

.transcript-view text {
    background-color: #181c26;
    color: inherit;
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

button.ready-btn {
    background-color: #182338;
    color: #8ab4ff;
    border: 1px solid #4f8cff;
    border-radius: 8px;
}

button.ready-btn:hover {
    background-color: #1f2e4a;
}

.queue-pill {
    background-color: #1d2b49;
    color: #8ab4ff;
    border: 1px solid #355796;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
}

.engine-pill {
    background-color: #1d3036;
    color: #9bd5de;
    border: 1px solid #355a62;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 700;
}

.pro-state {
    background-color: #252b37;
    color: #aab5c8;
    border: 1px solid #394253;
    border-radius: 999px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 700;
}

.pro-state.busy {
    background-color: #382f1d;
    color: #f0cf82;
    border-color: #65512b;
}

.pro-state.ready {
    background-color: #193326;
    color: #8de0aa;
    border-color: #2b6542;
}

.pro-state.error {
    background-color: #382226;
    color: #f2a0a8;
    border-color: #6a3941;
}

.backend-pill {
    color: #8b95a8;
    font-size: 12px;
    padding: 2px 8px;
}

.backend-pill.ready {
    color: #6dd58c;
}
"""


class TranscriptionWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title=APP_TITLE)
        self.set_default_size(920, 800)
        self.set_size_request(680, 580)

        self._config: dict[str, Any] = engine.load_config()
        self._presets: list[engine.ContextPreset] = engine.get_all_presets(self._config)
        self._restoring_ui = False

        self._last_state: dict[str, Any] = {}
        self._has_transcript = False
        self._closing_confirmed = False
        self._pulse_source_id: int | None = None
        self._timer_source_id: int | None = None
        self._copy_reset_source_id: int | None = None
        self._save_prompt_source_id: int | None = None
        self._pro_checking = False
        self._pro_backend: Any = None
        self._pro_choices: list[Any] = []
        self._pending_pro_paths: list[Path] = []
        self._pro_cleanup_pending = False
        self._pro_cleanup_done = False

        self._worker = engine.TranscriptionWorker(self._on_worker_update)
        self._live_transcriber: Any = None

        self._build_ui()
        self._restore_saved_settings()
        if self._pro_switch.get_active():
            self._check_pro_model_async()
        self.connect("close-request", self._on_close_request)

    def _build_ui(self) -> None:
        root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Header Bar
        header = Adw.HeaderBar()
        title_widget = Adw.WindowTitle(
            title="Trascrizione Audio Locale",
            subtitle="GPU CUDA / CPU + VAD • Salvataggio automatico in Trascrizioni/",
        )
        header.set_title_widget(title_widget)

        self._backend_label = Gtk.Label(label="CUDA in avvio...")
        self._backend_label.add_css_class("backend-pill")
        header.pack_end(self._backend_label)

        root_box.append(header)

        # Toast overlay wrapping main workspace
        self._toast_overlay = Adw.ToastOverlay()
        self._toast_overlay.set_vexpand(True)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(20)
        content.set_margin_end(20)
        content.set_vexpand(True)

        # 1. Controls Card
        controls_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        controls_card.add_css_class("card-panel")

        # Row 1: Model + Language
        row_top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)

        model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        model_box.set_hexpand(True)
        model_lbl = Gtk.Label(label="Modello Whisper", xalign=0.0)
        model_lbl.add_css_class("field-label")
        self._model_dropdown = Gtk.DropDown.new_from_strings(
            [label for _, label in engine.MODEL_OPTIONS]
        )
        self._model_dropdown.set_selected(0)
        self._model_dropdown.set_hexpand(True)
        self._model_dropdown.connect("notify::selected", self._on_model_changed)
        model_box.append(model_lbl)
        model_box.append(self._model_dropdown)

        lang_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        lang_lbl = Gtk.Label(label="Lingua", xalign=0.0)
        lang_lbl.add_css_class("field-label")
        self._lang_dropdown = Gtk.DropDown.new_from_strings(
            [label for _, label in engine.LANGUAGE_OPTIONS]
        )
        self._lang_dropdown.set_selected(0)
        self._lang_dropdown.set_size_request(185, -1)
        self._lang_dropdown.connect("notify::selected", self._on_lang_changed)
        lang_box.append(lang_lbl)
        lang_box.append(self._lang_dropdown)

        diar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        diar_lbl = Gtk.Label(label="Riconoscimento voci", xalign=0.0)
        diar_lbl.add_css_class("field-label")
        self._diar_dropdown = Gtk.DropDown.new_from_strings(
            [label for _, label in engine.DIARIZATION_OPTIONS]
        )
        self._diar_dropdown.set_selected(0)
        self._diar_dropdown.set_size_request(210, -1)
        self._diar_dropdown.connect("notify::selected", self._on_diar_changed)
        diar_box.append(diar_lbl)
        diar_box.append(self._diar_dropdown)

        row_top.append(model_box)
        row_top.append(lang_box)
        row_top.append(diar_box)
        controls_card.append(row_top)

        # Row 2: Context preset dropdown + Show timestamps
        row_ctx = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)

        ctx_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        ctx_box.set_hexpand(True)
        ctx_lbl = Gtk.Label(label="Contesto / Profilo", xalign=0.0)
        ctx_lbl.add_css_class("field-label")
        self._preset_dropdown = Gtk.DropDown.new_from_strings(
            [p.name for p in self._presets]
        )
        self._preset_dropdown.set_hexpand(True)
        self._preset_dropdown.connect("notify::selected", self._on_preset_changed)
        ctx_box.append(ctx_lbl)
        ctx_box.append(self._preset_dropdown)

        self._show_ts_check = Gtk.CheckButton(label="Mostra timestamp")
        self._show_ts_check.set_active(False)
        self._show_ts_check.set_valign(Gtk.Align.END)
        self._show_ts_check.set_margin_bottom(6)
        self._show_ts_check.connect("toggled", self._on_show_ts_toggled)

        row_ctx.append(ctx_box)
        row_ctx.append(self._show_ts_check)
        controls_card.append(row_ctx)

        pro_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        pro_info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        pro_heading = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        pro_badge = Gtk.Label(label="PRO")
        pro_badge.add_css_class("queue-pill")
        pro_label = Gtk.Label(label="Correzione con LLM locale", xalign=0.0)
        pro_label.add_css_class("field-label")
        self._pro_status_label = Gtk.Label(label="● Spento")
        self._pro_status_label.add_css_class("pro-state")
        pro_heading.append(pro_badge)
        pro_heading.append(pro_label)
        pro_heading.append(self._pro_status_label)
        pro_info.append(pro_heading)
        self._pro_hint = Gtk.Label(
            label=(
                "LM Studio e Ollama avviano i servizi in background; se trovi "
                "più modelli puoi scegliere quale usare."
            ),
            xalign=0.0,
        )
        self._pro_hint.add_css_class("hint-label")
        self._pro_hint.set_wrap(True)
        pro_info.append(self._pro_hint)
        pro_info.set_hexpand(True)
        pro_row.append(pro_info)
        self._pro_release_btn = Gtk.Button(label="Libera memoria")
        self._pro_release_btn.set_sensitive(False)
        self._pro_release_btn.set_tooltip_text(
            "Scarica il modello e disattiva PRO; riattivalo per caricarlo di nuovo"
        )
        self._pro_release_btn.connect("clicked", self._on_pro_release_clicked)
        pro_row.append(self._pro_release_btn)
        self._pro_switch = Gtk.Switch()
        self._pro_switch.set_valign(Gtk.Align.CENTER)
        self._pro_switch.set_tooltip_text(
            "Corregge e migliora la leggibilità dopo la trascrizione"
        )
        self._pro_switch.connect("notify::active", self._on_pro_switch_changed)
        pro_row.append(self._pro_switch)
        controls_card.append(pro_row)

        # Row 4: Collapsible prompt editor & custom profile manager
        self._prompt_expander = Gtk.Expander(label="Personalizza prompt di contesto")
        self._prompt_expander.set_expanded(False)

        prompt_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        prompt_panel.set_margin_top(8)

        prompt_hint = Gtk.Label(
            label="Inserisci argomento, sigle e termini tecnici (nella stessa lingua dell'audio) per guidare il riconoscimento.",
            xalign=0.0,
        )
        prompt_hint.add_css_class("hint-label")
        prompt_hint.set_wrap(True)
        prompt_panel.append(prompt_hint)

        prompt_scroll = Gtk.ScrolledWindow()
        prompt_scroll.add_css_class("prompt-scroll")
        prompt_scroll.set_min_content_height(84)
        prompt_scroll.set_max_content_height(130)
        prompt_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._prompt_view = Gtk.TextView()
        self._prompt_view.add_css_class("prompt-view")
        self._prompt_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._prompt_view.set_top_margin(8)
        self._prompt_view.set_bottom_margin(8)
        self._prompt_view.set_left_margin(10)
        self._prompt_view.set_right_margin(10)
        self._prompt_buffer = self._prompt_view.get_buffer()
        self._prompt_buffer.connect("changed", self._on_prompt_text_changed)

        prompt_scroll.set_child(self._prompt_view)
        prompt_panel.append(prompt_scroll)

        preset_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        self._preset_name_entry = Gtk.Entry()
        self._preset_name_entry.set_placeholder_text(
            "Nome profilo (es. Cardiologia IT, Meccanica Quantistica EN)..."
        )
        self._preset_name_entry.set_hexpand(True)
        self._preset_name_entry.connect("activate", self._on_save_preset_clicked)

        self._save_preset_btn = Gtk.Button(label="Salva profilo")
        self._save_preset_btn.add_css_class("accent-btn")
        self._save_preset_btn.connect("clicked", self._on_save_preset_clicked)

        self._reset_prompt_btn = Gtk.Button(label="Ripristina")
        self._reset_prompt_btn.add_css_class("secondary-btn")
        self._reset_prompt_btn.set_tooltip_text(
            "Ripristina il testo salvato del profilo selezionato"
        )
        self._reset_prompt_btn.connect("clicked", self._on_reset_prompt_clicked)

        self._delete_preset_btn = Gtk.Button(label="Elimina")
        self._delete_preset_btn.add_css_class("destructive-action")
        self._delete_preset_btn.set_sensitive(False)
        self._delete_preset_btn.set_tooltip_text(
            "Elimina il profilo personalizzato selezionato"
        )
        self._delete_preset_btn.connect("clicked", self._on_delete_preset_clicked)

        preset_actions.append(self._preset_name_entry)
        preset_actions.append(self._save_preset_btn)
        preset_actions.append(self._reset_prompt_btn)
        preset_actions.append(self._delete_preset_btn)
        prompt_panel.append(preset_actions)

        self._prompt_expander.set_child(prompt_panel)
        controls_card.append(self._prompt_expander)

        content.append(controls_card)

        # 2. Drag & Drop + Browse Zone
        self._drop_zone = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self._drop_zone.add_css_class("drop-zone")
        self._drop_zone.set_halign(Gtk.Align.FILL)
        self._drop_zone.set_cursor_from_name("pointer")

        drop_icon = Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic")
        drop_icon.set_pixel_size(32)
        drop_icon.set_opacity(0.85)

        drop_title = Gtk.Label(label="Trascina qui uno o più file audio o video")
        drop_title.add_css_class("drop-title")

        drop_sub = Gtk.Label(
            label="oppure clicca per sceglierli (.mp3, .wav, .m4a, .mp4, .mkv, .ogg, .flac, .webm) • i file extra vengono messi in coda"
        )
        drop_sub.add_css_class("drop-subtitle")
        drop_sub.set_wrap(True)
        drop_sub.set_justify(Gtk.Justification.CENTER)

        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        btn_row.set_halign(Gtk.Align.CENTER)
        btn_row.set_margin_top(6)

        browse_btn = Gtk.Button(label="Sfoglia file...")
        browse_btn.add_css_class("accent-btn")
        browse_btn.connect("clicked", lambda _btn: self._open_file_dialog())

        self._live_btn = Gtk.Button(label="LIVE (Microfono)")
        self._live_btn.add_css_class("secondary-btn")
        self._live_btn.connect("clicked", self._on_live_clicked)

        btn_row.append(browse_btn)
        btn_row.append(self._live_btn)

        self._drop_zone.append(drop_icon)
        self._drop_zone.append(drop_title)
        self._drop_zone.append(drop_sub)
        self._drop_zone.append(btn_row)

        # Clicking anywhere inside the drop zone opens the file picker
        click_gesture = Gtk.GestureClick()
        click_gesture.connect("released", self._on_drop_zone_clicked)
        self._drop_zone.add_controller(click_gesture)

        # Native Wayland Drag & Drop on both drop zone and entire window
        self._attach_drop_target(self._drop_zone)
        self._attach_drop_target(content)

        content.append(self._drop_zone)

        # 3. Progress Bar & Metadata Row
        progress_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_fraction(0.0)
        progress_box.append(self._progress_bar)

        meta_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        self._engine_badge = Gtk.Label(label="")
        self._engine_badge.add_css_class("engine-pill")
        self._engine_badge.set_visible(False)

        self._status_label = Gtk.Label(
            label="Pronto. Trascina uno o più file audio o video.", xalign=0.0
        )
        self._status_label.add_css_class("muted-label")
        self._status_label.set_ellipsize(3)  # Pango.EllipsizeMode.END
        self._status_label.set_hexpand(True)

        self._queue_badge = Gtk.Label(label="")
        self._queue_badge.add_css_class("queue-pill")
        self._queue_badge.set_visible(False)

        self._info_label = Gtk.Label(label="", xalign=1.0)
        self._info_label.add_css_class("muted-label")

        meta_row.append(self._engine_badge)
        meta_row.append(self._status_label)
        meta_row.append(self._queue_badge)
        meta_row.append(self._info_label)
        progress_box.append(meta_row)

        content.append(progress_box)

        # 4. Transcript Viewer
        self._scroll = Gtk.ScrolledWindow()
        self._scroll.add_css_class("transcript-scroll")
        self._scroll.set_vexpand(True)
        self._scroll.set_hexpand(True)
        self._scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._text_view = Gtk.TextView()
        self._text_view.add_css_class("transcript-view")
        self._text_view.add_css_class("placeholder-text")
        self._text_view.set_editable(False)
        self._text_view.set_cursor_visible(False)
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._text_view.set_top_margin(16)
        self._text_view.set_bottom_margin(16)
        self._text_view.set_left_margin(18)
        self._text_view.set_right_margin(18)
        self._text_view.set_pixels_below_lines(6)
        self._text_view.set_pixels_inside_wrap(3)

        self._text_buffer = self._text_view.get_buffer()
        self._text_buffer.set_text("Il testo trascritto apparirà qui in tempo reale...")

        self._scroll.set_child(self._text_view)
        content.append(self._scroll)

        # 5. Bottom Action Bar
        actions_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        self._copy_btn = Gtk.Button(label="Copia testo")
        self._copy_btn.add_css_class("secondary-btn")
        self._copy_btn.set_sensitive(False)
        self._copy_btn.connect("clicked", self._on_copy_clicked)

        self._open_txt_btn = Gtk.Button(label="Apri .txt")
        self._open_txt_btn.add_css_class("secondary-btn")
        self._open_txt_btn.set_sensitive(False)
        self._open_txt_btn.connect("clicked", self._on_open_txt_clicked)

        self._rename_voices_btn = Gtk.Button(label="Rinomina voci")
        self._rename_voices_btn.add_css_class("secondary-btn")
        self._rename_voices_btn.set_visible(False)
        self._rename_voices_btn.set_sensitive(False)
        self._rename_voices_btn.connect("clicked", self._on_rename_voices_clicked)

        self._open_dir_btn = Gtk.Button(label="Apri cartella Trascrizioni")
        self._open_dir_btn.add_css_class("secondary-btn")
        self._open_dir_btn.connect("clicked", self._on_open_dir_clicked)

        actions_spacer = Gtk.Box()
        actions_spacer.set_hexpand(True)

        self._cancel_btn = Gtk.Button(label="Annulla")
        self._cancel_btn.add_css_class("destructive-action")
        self._cancel_btn.set_visible(False)
        self._cancel_btn.connect("clicked", self._on_cancel_clicked)

        actions_row.append(self._copy_btn)
        actions_row.append(self._open_txt_btn)
        actions_row.append(self._rename_voices_btn)
        actions_row.append(self._open_dir_btn)
        actions_row.append(actions_spacer)
        actions_row.append(self._cancel_btn)

        content.append(actions_row)

        self._toast_overlay.set_child(content)
        root_box.append(self._toast_overlay)
        self.set_content(root_box)

    # --- Settings & Context Preset management ---

    def _restore_saved_settings(self) -> None:
        self._restoring_ui = True
        try:
            self._set_model_by_key(self._config.get("model", "large-v3"))
            self._set_preset_by_id(self._config.get("preset_id", "none"))
            self._set_language_by_key(self._config.get("language", "auto"))
            self._set_diarization_by_key(self._config.get("diarization", "off"))

            preset = self._selected_preset()
            saved_prompt = self._config.get("custom_prompt")
            prompt_to_show = (
                saved_prompt if isinstance(saved_prompt, str) and saved_prompt else preset.prompt
            )
            if preset.id == "none" and not saved_prompt:
                prompt_to_show = ""
            self._prompt_buffer.set_text(prompt_to_show)
            self._delete_preset_btn.set_sensitive(not preset.builtin)
            self._preset_name_entry.set_text("" if preset.builtin else preset.name)
            self._show_ts_check.set_active(bool(self._config.get("show_timestamps", False)))
            self._pro_switch.set_active(bool(self._config.get("pro_enabled", False)))
        finally:
            self._restoring_ui = False

    def _rebuild_preset_dropdown(self, select_id: str) -> None:
        self._presets = engine.get_all_presets(self._config)
        self._restoring_ui = True
        try:
            model = Gtk.StringList.new([p.name for p in self._presets])
            self._preset_dropdown.set_model(model)
            self._set_preset_by_id(select_id)
            preset = self._selected_preset()
            self._delete_preset_btn.set_sensitive(not preset.builtin)
            self._preset_name_entry.set_text("" if preset.builtin else preset.name)
        finally:
            self._restoring_ui = False

    def _set_model_by_key(self, key: str) -> None:
        for idx, (k, _) in enumerate(engine.MODEL_OPTIONS):
            if k == key:
                self._model_dropdown.set_selected(idx)
                return
        self._model_dropdown.set_selected(0)

    def _set_language_by_key(self, key: str) -> None:
        for idx, (k, _) in enumerate(engine.LANGUAGE_OPTIONS):
            if k == key:
                self._lang_dropdown.set_selected(idx)
                return
        self._lang_dropdown.set_selected(0)

    def _set_diarization_by_key(self, key: str) -> None:
        for idx, (k, _) in enumerate(engine.DIARIZATION_OPTIONS):
            if k == key:
                self._diar_dropdown.set_selected(idx)
                return
        self._diar_dropdown.set_selected(0)

    def _set_preset_by_id(self, preset_id: str) -> None:
        for idx, preset in enumerate(self._presets):
            if preset.id == preset_id:
                self._preset_dropdown.set_selected(idx)
                return
        self._preset_dropdown.set_selected(0)

    def _selected_preset(self) -> engine.ContextPreset:
        idx = self._preset_dropdown.get_selected()
        if 0 <= idx < len(self._presets):
            return self._presets[idx]
        return self._presets[0]

    def _current_prompt_text(self) -> str:
        start = self._prompt_buffer.get_start_iter()
        end = self._prompt_buffer.get_end_iter()
        return self._prompt_buffer.get_text(start, end, False).strip()

    def _persist_current_config(self) -> None:
        self._config["model"] = self._selected_model()
        self._config["language"] = self._selected_language()
        self._config["diarization"] = self._selected_diarization()
        self._config["preset_id"] = self._selected_preset().id
        self._config["custom_prompt"] = self._current_prompt_text()
        self._config["show_timestamps"] = self._show_ts_check.get_active()
        self._config["pro_enabled"] = self._pro_switch.get_active()
        try:
            engine.save_config(self._config)
        except OSError:
            pass

    def _on_model_changed(self, _dropdown: Gtk.DropDown, _pspec: Any) -> None:
        if not self._restoring_ui:
            self._persist_current_config()

    def _on_lang_changed(self, _dropdown: Gtk.DropDown, _pspec: Any) -> None:
        if not self._restoring_ui:
            self._persist_current_config()

    def _on_diar_changed(self, _dropdown: Gtk.DropDown, _pspec: Any) -> None:
        if not self._restoring_ui:
            self._persist_current_config()

    def _on_show_ts_toggled(self, _btn: Gtk.CheckButton) -> None:
        self._render_transcript()
        if not self._restoring_ui:
            self._persist_current_config()

    def _on_pro_switch_changed(self, switch: Gtk.Switch, _pspec: Any) -> None:
        if self._restoring_ui:
            return
        if switch.get_active():
            self._check_pro_model_async()
        else:
            if self._pro_backend is not None and self._worker.is_busy():
                self._restoring_ui = True
                switch.set_active(True)
                self._restoring_ui = False
                self._show_toast(
                    "Attendi la fine dell'operazione prima di disattivare PRO."
                )
                return
            self._config["pro_enabled"] = False
            self._persist_current_config()
            if self._pro_backend is not None:
                self._on_pro_release_clicked(self._pro_release_btn)
            else:
                self._set_pro_status("● Spento")

    def _set_pro_status(self, label: str, state: str = "") -> None:
        self._pro_status_label.set_text(label)
        for css_class in ("busy", "ready", "error"):
            self._pro_status_label.remove_css_class(css_class)
        if state:
            self._pro_status_label.add_css_class(state)

    def _show_pro_activity(self, message: str) -> None:
        if self._worker.is_busy():
            return
        self._engine_badge.set_text("LLM locale")
        self._engine_badge.set_visible(True)
        self._status_label.set_text(message)
        self._info_label.set_text("Stima: non disponibile in questa fase")
        self._progress_bar.set_fraction(0.0)
        self._start_pulse()

    def _finish_pro_activity(self, message: str, detail: str = "") -> None:
        if self._worker.is_busy():
            return
        self._stop_pulse()
        self._progress_bar.set_fraction(0.0)
        self._engine_badge.set_visible(False)
        self._status_label.set_text(message)
        self._info_label.set_text(detail)

    def _on_pro_release_clicked(self, _button: Gtk.Button) -> None:
        if self._pro_checking or self._worker.is_busy():
            self._show_toast("Attendi la fine dell'operazione PRO prima di liberare memoria.")
            return
        self._pro_release_btn.set_sensitive(False)
        self._pro_switch.set_sensitive(False)
        self._set_pro_status("● Rilascio memoria…", "busy")
        self._show_pro_activity("LLM locale: rilascio memoria...")
        threading.Thread(target=self._release_pro_model, daemon=True).start()

    def _release_pro_model(self) -> None:
        error = ""
        try:
            import pro

            pro.release_local_model()
        except Exception as exc:
            error = str(exc)
        GLib.idle_add(self._finish_pro_release, error)

    def _finish_pro_release(self, error: str) -> bool:
        if self._pro_cleanup_pending or self._pro_cleanup_done:
            return GLib.SOURCE_REMOVE
        self._pro_switch.set_sensitive(True)
        if error:
            self._pro_release_btn.set_sensitive(self._pro_backend is not None)
            self._set_pro_status("● Errore rilascio", "error")
            self._pro_hint.set_text(error)
            self._finish_pro_activity("LLM locale: errore durante il rilascio.")
            self._show_toast(f"Errore rilascio risorse PRO: {error}")
        else:
            self._pro_backend = None
            self._pro_release_btn.set_sensitive(False)
            self._pro_switch.set_active(False)
            self._set_pro_status("● Memoria liberata")
            self._pro_hint.set_text("Riattiva PRO per caricare di nuovo un modello.")
            self._finish_pro_activity("LLM locale: memoria liberata.")
            self._show_toast(
                "Memoria liberata e PRO disattivato. Riattiva PRO per usare di nuovo il modello."
            )
        return GLib.SOURCE_REMOVE

    def _check_pro_model_async(self) -> None:
        self._pro_checking = True
        self._pro_switch.set_sensitive(False)
        self._pro_release_btn.set_sensitive(False)
        self._set_pro_status("● Ricerca modelli…", "busy")
        self._pro_hint.set_text("Ricerca dei modelli chat installati in locale.")
        self._show_pro_activity("LLM locale: ricerca modelli installati...")
        threading.Thread(target=self._discover_pro_models, daemon=True).start()

    def _discover_pro_models(self) -> None:
        try:
            import pro

            models = pro.discover_local_models()
            GLib.idle_add(self._finish_pro_discovery, models, "")
        except Exception as exc:
            GLib.idle_add(self._finish_pro_discovery, [], str(exc))

    def _finish_pro_discovery(self, models: list[Any], error: str) -> bool:
        if self._pro_cleanup_pending or self._pro_cleanup_done:
            return GLib.SOURCE_REMOVE
        if error:
            self._finish_pro_model_check(None, error)
        elif not models:
            self._finish_pro_model_check(None, "Nessun modello chat locale trovato.")
        elif len(models) == 1:
            self._start_pro_model_async(models[0])
        else:
            self._pro_choices = models
            self._set_pro_status("● Scegli modello", "busy")
            self._pro_hint.set_text(f"Trovati {len(models)} modelli chat locali.")
            self._finish_pro_activity("Seleziona il modello PRO da caricare.")
            self._show_pro_model_picker(models)
        return GLib.SOURCE_REMOVE

    def _show_pro_model_picker(self, models: list[Any]) -> None:
        dialog = Adw.MessageDialog.new(
            self,
            "Scegli modello PRO",
            "Sono installati più modelli locali. Seleziona quello da usare:",
        )
        dropdown = Gtk.DropDown.new_from_strings([model.label for model in models])
        dropdown.set_selected(0)
        dropdown.set_hexpand(True)
        dialog.set_extra_child(dropdown)
        dialog.add_response("select", "Usa modello")
        dialog.add_response("cancel", "Annulla")
        dialog.set_default_response("select")
        dialog.set_close_response("cancel")
        dialog.set_response_appearance("select", Adw.ResponseAppearance.SUGGESTED)
        dialog.connect("response", self._on_pro_model_choice, dropdown)
        dialog.present()

    def _on_pro_model_choice(
        self, _dialog: Adw.MessageDialog, response: str, dropdown: Gtk.DropDown
    ) -> None:
        if self._pro_cleanup_pending or self._pro_cleanup_done:
            return
        if response == "cancel":
            self._cancel_pro_activation()
            return
        index = dropdown.get_selected()
        if not 0 <= index < len(self._pro_choices):
            self._cancel_pro_activation()
            return
        model = self._pro_choices[index]
        self._pro_choices = []
        self._start_pro_model_async(model)

    def _cancel_pro_activation(self) -> None:
        self._pro_checking = False
        self._pro_backend = None
        self._pro_choices = []
        self._pro_switch.set_sensitive(True)
        self._pro_release_btn.set_sensitive(False)
        self._set_pro_status("● Spento")
        self._pro_hint.set_text("Attiva PRO per cercare i modelli locali installati.")
        self._finish_pro_activity("PRO disattivato.")
        self._restoring_ui = True
        self._pro_switch.set_active(False)
        self._restoring_ui = False
        self._config["pro_enabled"] = False
        self._persist_current_config()
        threading.Thread(
            target=self._release_discovered_pro_resources, daemon=True
        ).start()
        self._enqueue_pending_pro_paths()

    def _release_discovered_pro_resources(self) -> None:
        try:
            import pro

            pro.release_local_model()
        except Exception as exc:
            GLib.idle_add(self._show_toast, f"Errore rilascio risorse PRO: {exc}")

    def _start_pro_model_async(self, backend: Any) -> None:
        self._pro_backend = backend
        self._set_pro_status("● Caricamento…", "busy")
        self._pro_hint.set_text(backend.label)
        self._show_pro_activity(f"LLM locale: caricamento {backend.label}...")
        threading.Thread(
            target=self._load_pro_model, args=(backend,), daemon=True
        ).start()

    def _load_pro_model(self, backend: Any) -> None:
        try:
            import pro

            pro.load_local_model(
                backend,
                progress=lambda message: GLib.idle_add(
                    self._status_label.set_text, message
                ),
            )
            GLib.idle_add(self._finish_pro_model_check, backend, "")
        except Exception as exc:
            try:
                import pro

                pro.release_local_model()
            except Exception as cleanup_error:
                exc = RuntimeError(f"{exc}; {cleanup_error}")
            GLib.idle_add(self._finish_pro_model_check, None, str(exc))

    def _finish_pro_model_check(self, backend: Any, error: str) -> bool:
        if self._pro_cleanup_pending or self._pro_cleanup_done:
            return GLib.SOURCE_REMOVE
        self._pro_checking = False
        self._pro_switch.set_sensitive(True)
        if error:
            self._pro_backend = None
            self._pro_release_btn.set_sensitive(False)
            self._set_pro_status("● Modello non disponibile", "error")
            self._pro_hint.set_text(error)
            self._restoring_ui = True
            self._pro_switch.set_active(False)
            self._restoring_ui = False
            self._config["pro_enabled"] = False
            self._persist_current_config()
            self._finish_pro_activity("LLM locale: caricamento non riuscito.")
            self._show_pro_model_dialog(error)
            self._enqueue_pending_pro_paths()
        else:
            self._pro_backend = backend
            self._pro_release_btn.set_sensitive(True)
            self._set_pro_status("● Modello caricato", "ready")
            self._pro_hint.set_text(
                f"{backend.label} è pronto per correggere le trascrizioni."
            )
            self._config["pro_enabled"] = True
            self._persist_current_config()
            self._finish_pro_activity(
                f"LLM locale pronto: {backend.label}", "Modello caricato"
            )
            self._show_toast(f"PRO attivo: {backend.label}")
            self._enqueue_pending_pro_paths()
        return GLib.SOURCE_REMOVE

    def _enqueue_pending_pro_paths(self) -> None:
        paths, self._pending_pro_paths = self._pending_pro_paths, []
        if paths:
            if not self._pro_switch.get_active():
                self._show_toast("PRO non disponibile; avvio trascrizione senza correzione.")
            self.enqueue_paths(paths)

    def _show_pro_model_dialog(self, error: str) -> None:
        import pro

        dialog = Adw.MessageDialog.new(
            self,
            "Modello PRO non disponibile",
            (
                f"{error}\n\nInstalla o carica un modello locale in Ollama o "
                "LM Studio. La guida contiene i passaggi per configurarlo."
            ),
        )
        dialog.add_response("close", "Chiudi")
        dialog.add_response("guide", "Apri guida installazione")
        dialog.set_default_response("guide")
        dialog.set_close_response("close")
        dialog.set_response_appearance(
            "guide", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.connect("response", self._on_pro_dialog_response)
        dialog.present()

    def _on_pro_dialog_response(
        self, _dialog: Adw.MessageDialog, response: str
    ) -> None:
        if response == "guide":
            import pro

            try:
                Gio.AppInfo.launch_default_for_uri(pro.GUIDE_URL, None)
            except GLib.Error as error:
                self._show_toast(f"Impossibile aprire la guida: {error}")

    def _on_preset_changed(self, _dropdown: Gtk.DropDown, _pspec: Any) -> None:
        if self._restoring_ui:
            return
        preset = self._selected_preset()
        self._restoring_ui = True
        try:
            self._prompt_buffer.set_text(preset.prompt)
            self._set_language_by_key(preset.language)
            self._delete_preset_btn.set_sensitive(not preset.builtin)
            self._preset_name_entry.set_text("" if preset.builtin else preset.name)
        finally:
            self._restoring_ui = False
        self._persist_current_config()

    def _on_prompt_text_changed(self, _buf: Gtk.TextBuffer) -> None:
        if self._restoring_ui:
            return
        if self._save_prompt_source_id is not None:
            GLib.source_remove(self._save_prompt_source_id)
        self._save_prompt_source_id = GLib.timeout_add(
            400, self._flush_prompt_save
        )

    def _flush_prompt_save(self) -> bool:
        self._save_prompt_source_id = None
        self._persist_current_config()
        return GLib.SOURCE_REMOVE

    def _on_save_preset_clicked(self, _widget: Gtk.Widget) -> None:
        name = self._preset_name_entry.get_text().strip()
        if not name:
            self._show_toast("Inserisci un nome per salvare il profilo.")
            self._preset_name_entry.grab_focus()
            return
        prompt = self._current_prompt_text()
        language = self._selected_language()
        try:
            updated_cfg, preset_id = engine.upsert_user_preset(
                self._config, name=name, language=language, prompt=prompt
            )
        except ValueError as exc:
            self._show_toast(str(exc))
            return
        self._config = updated_cfg
        self._rebuild_preset_dropdown(preset_id)
        self._persist_current_config()
        self._show_toast(f"Profilo salvato: {name}")

    def _on_reset_prompt_clicked(self, _btn: Gtk.Button) -> None:
        preset = self._selected_preset()
        self._restoring_ui = True
        try:
            self._prompt_buffer.set_text(preset.prompt)
            self._set_language_by_key(preset.language)
        finally:
            self._restoring_ui = False
        self._persist_current_config()
        self._show_toast(f"Ripristinato il testo del profilo \"{preset.name}\"")

    def _on_delete_preset_clicked(self, _btn: Gtk.Button) -> None:
        preset = self._selected_preset()
        if preset.builtin:
            return
        deleted_name = preset.name
        self._config = engine.delete_user_preset(self._config, preset.id)
        self._rebuild_preset_dropdown("none")
        self._restoring_ui = True
        try:
            self._prompt_buffer.set_text("")
            self._set_language_by_key("auto")
        finally:
            self._restoring_ui = False
        self._persist_current_config()
        self._show_toast(f"Profilo eliminato: {deleted_name}")

    # --- Backend & Drag/Drop ---

    def set_backend_status(self, text: str, ready: bool = False) -> None:
        self._backend_label.set_text(text)
        if ready:
            self._backend_label.add_css_class("ready")
        else:
            self._backend_label.remove_css_class("ready")

    def _attach_drop_target(self, widget: Gtk.Widget) -> None:
        drop_target = Gtk.DropTarget.new(Gdk.FileList.__gtype__, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gdk.FileList.__gtype__, Gio.File.__gtype__])
        drop_target.connect("enter", self._on_drag_enter)
        drop_target.connect("leave", self._on_drag_leave)
        drop_target.connect("drop", self._on_drop)
        widget.add_controller(drop_target)

    def _on_drag_enter(
        self, _target: Gtk.DropTarget, _x: float, _y: float
    ) -> Gdk.DragAction:
        self._drop_zone.add_css_class("drop-over")
        return Gdk.DragAction.COPY

    def _on_drag_leave(self, _target: Gtk.DropTarget) -> None:
        self._drop_zone.remove_css_class("drop-over")

    def _on_drop(
        self, _target: Gtk.DropTarget, value: Any, _x: float, _y: float
    ) -> bool:
        self._drop_zone.remove_css_class("drop-over")
        paths: list[Path] = []
        if isinstance(value, Gdk.FileList):
            for gfile in value.get_files():
                p = gfile.get_path()
                if p:
                    paths.append(Path(p))
        elif isinstance(value, Gio.File):
            p = value.get_path()
            if p:
                paths.append(Path(p))
        if not paths:
            return False
        self.enqueue_paths(paths)
        return True

    def _on_drop_zone_clicked(
        self, _gesture: Gtk.GestureClick, _n_press: int, x: float, y: float
    ) -> None:
        picked = self._drop_zone.pick(x, y, Gtk.PickFlags.DEFAULT)
        if picked is not None and picked.get_ancestor(Gtk.Button) is not None:
            return
        self._open_file_dialog()

    def _open_file_dialog(self) -> None:
        dialog = Gtk.FileDialog()
        dialog.set_title("Seleziona uno o più file audio o video")

        media_filter = Gtk.FileFilter()
        media_filter.set_name("File Audio e Video")
        media_filter.add_mime_type("audio/*")
        media_filter.add_mime_type("video/*")
        for ext in sorted(engine.SUPPORTED_EXTENSIONS):
            media_filter.add_pattern(f"*{ext}")
            media_filter.add_pattern(f"*{ext.upper()}")

        all_filter = Gtk.FileFilter()
        all_filter.set_name("Tutti i file")
        all_filter.add_pattern("*")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(media_filter)
        filters.append(all_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(media_filter)

        dialog.open_multiple(self, None, self._on_file_dialog_response)

    def _on_file_dialog_response(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        try:
            files_model = dialog.open_multiple_finish(result)
        except GLib.Error:
            return
        if files_model is None:
            return
        paths: list[Path] = []
        for idx in range(files_model.get_n_items()):
            gfile = files_model.get_item(idx)
            if isinstance(gfile, Gio.File):
                p = gfile.get_path()
                if p:
                    paths.append(Path(p))
        if paths:
            self.enqueue_paths(paths)

    def _selected_model(self) -> str:
        idx = self._model_dropdown.get_selected()
        if 0 <= idx < len(engine.MODEL_OPTIONS):
            return engine.MODEL_OPTIONS[idx][0]
        return "large-v3"

    def _selected_language(self) -> str:
        idx = self._lang_dropdown.get_selected()
        if 0 <= idx < len(engine.LANGUAGE_OPTIONS):
            return engine.LANGUAGE_OPTIONS[idx][0]
        return "auto"

    def _selected_diarization(self) -> str:
        idx = self._diar_dropdown.get_selected()
        if 0 <= idx < len(engine.DIARIZATION_OPTIONS):
            return engine.DIARIZATION_OPTIONS[idx][0]
        return "off"

    def enqueue_paths(self, paths: list[Path]) -> None:
        valid_files = [p for p in paths if p.is_file()]
        skipped = len(paths) - len(valid_files)
        if not valid_files:
            self._show_toast("Seleziona almeno un file audio o video valido.")
            return
        if self._pro_checking:
            self._pending_pro_paths.extend(valid_files)
            self._show_toast("File in coda, in attesa della scelta del modello PRO.")
            return

        was_busy = self._worker.is_busy()
        model_mode = self._selected_model()
        language = self._selected_language()
        diarization_mode = self._selected_diarization()
        initial_prompt = self._current_prompt_text()

        jobs = [
            engine.TranscriptionJob(
                file_path=p,
                language=language,
                model_mode=model_mode,
                initial_prompt=initial_prompt,
                diarization_mode=diarization_mode,
                pro_enabled=self._pro_switch.get_active(),
                pro_provider=(
                    self._pro_backend.provider if self._pro_backend else ""
                ),
                pro_model=self._pro_backend.model if self._pro_backend else "",
            )
            for p in valid_files
        ]
        added = self._worker.enqueue(jobs)

        if added == 0:
            return
        if was_busy:
            msg = (
                f"Aggiunto alla coda: {valid_files[0].name}"
                if added == 1
                else f"Aggiunti {added} file alla coda"
            )
            self._show_toast(msg)
        elif added > 1:
            self._show_toast(
                f"Avviata trascrizione di {valid_files[0].name} (+{added - 1} in coda)"
            )
        if skipped > 0:
            self._show_toast(f"Ignorati {skipped} elementi non validi (es. cartelle).")

    # --- Worker UI Updates ---

    def _on_worker_update(self, snap: dict[str, Any]) -> None:
        GLib.idle_add(self._apply_state, snap)

    def _apply_state(self, snap: dict[str, Any]) -> bool:
        self._last_state = snap
        status = snap.get("status", "idle")
        progress = float(snap.get("progress", 0.0))

        engine_name = str(snap.get("engine", ""))
        self._engine_badge.set_text(engine_name)
        self._engine_badge.set_visible(
            bool(engine_name)
            and status in ("loading", "transcribing", "pro", "error")
        )
        queue_size = int(snap.get("queue_size", 0))

        self._status_label.set_text(snap.get("message", ""))

        if queue_size > 0:
            label = "In coda: 1 file" if queue_size == 1 else f"In coda: {queue_size} file"
            self._queue_badge.set_text(label)
            self._queue_badge.set_visible(True)
        else:
            self._queue_badge.set_visible(False)

        # Progress bar and timers
        if status == "loading" or (status == "pro" and progress <= 0):
            self._start_pulse()
            self._start_elapsed_timer()
            self._cancel_btn.set_visible(True)
            self._cancel_btn.set_sensitive(True)
        elif status in ("transcribing", "pro"):
            self._stop_pulse()
            self._progress_bar.set_fraction(max(0.0, min(1.0, progress / 100.0)))
            self._start_elapsed_timer()
            self._cancel_btn.set_visible(True)
            self._cancel_btn.set_sensitive(True)
        else:
            self._stop_pulse()
            self._stop_elapsed_timer()
            self._cancel_btn.set_visible(False)
            self._live_btn.set_sensitive(True)
            self._live_btn.set_label("LIVE (Microfono)")
            self._live_btn.remove_css_class("destructive-action")
            self._live_btn.add_css_class("secondary-btn")
            if status == "done":
                self._progress_bar.set_fraction(1.0)
            else:
                self._progress_bar.set_fraction(0.0)

        self._update_info_label(snap)

        # Update transcript viewer
        lines = snap.get("lines") or []
        if lines:
            self._has_transcript = True
            self._render_transcript(auto_scroll=True)
        elif status == "loading":
            self._has_transcript = False
            self._set_placeholder(f"Caricamento in corso per {snap.get('filename', '')}...")
        elif status == "error":
            self._has_transcript = False
            self._set_placeholder(f"Errore: {snap.get('error', '')}")
        elif status == "done" and not lines:
            self._has_transcript = False
            self._set_placeholder("(Nessun parlato rilevato nel file audio.)")

        if status == "error" and snap.get("pro_error"):
            self._show_pro_model_dialog(snap.get("error", "Errore LLM locale."))

        # Action buttons state
        self._copy_btn.set_sensitive(self._has_transcript)
        txt_ready = bool(snap.get("txt_path")) and Path(snap["txt_path"]).is_file()
        self._set_file_button_ready(self._open_txt_btn, txt_ready)

        speakers = snap.get("speakers") or []
        can_rename = status == "done" and len(speakers) > 0 and txt_ready
        self._rename_voices_btn.set_visible(can_rename)
        self._set_file_button_ready(self._rename_voices_btn, can_rename)

        if status == "done" and queue_size == 0:
            txt_name = Path(snap["txt_path"]).name if snap.get("txt_path") else ""
            if txt_name:
                self._show_toast(f"Salvato in Trascrizioni/{txt_name}")

        return GLib.SOURCE_REMOVE

    def _set_file_button_ready(self, button: Gtk.Button, ready: bool) -> None:
        button.set_sensitive(ready)
        if ready:
            button.remove_css_class("secondary-btn")
            button.add_css_class("ready-btn")
        else:
            button.remove_css_class("ready-btn")
            button.add_css_class("secondary-btn")

    def _update_info_label(self, snap: dict[str, Any]) -> None:
        parts: list[str] = []
        status = snap.get("status", "idle")
        progress = float(snap.get("progress", 0.0))
        if status == "transcribing" or (status == "pro" and progress > 0):
            parts.append(f"{progress:.0f}%")
        if snap.get("language"):
            parts.append(f"Lingua: {snap['language']}")
        duration = float(snap.get("duration", 0.0))
        if duration > 0:
            parts.append(f"Durata: {engine.format_short(duration)}")

        started_at = float(snap.get("job_started_at", 0.0))
        if status in ("loading", "transcribing", "pro") and started_at > 0:
            elapsed = max(0.0, time.time() - started_at)
        else:
            elapsed = float(snap.get("elapsed", 0.0))
        if elapsed > 0:
            parts.append(f"Tempo: {elapsed:.1f}s")

        progress_started_at = float(snap.get("progress_started_at") or 0.0)
        if status in ("loading", "transcribing", "pro"):
            if status == "loading" or (
                status == "pro" and progress_started_at <= 0
            ):
                parts.append("Stima: non disponibile in questa fase")
            elif progress < 5:
                parts.append("Stima: in calcolo")
            elif progress < 100:
                estimate_started_at = progress_started_at or started_at
                estimate_elapsed = max(0.0, time.time() - estimate_started_at)
                remaining = round(estimate_elapsed * (100.0 - progress) / progress)
                parts.append(f"Stima: ~{engine.format_short(remaining)}")

        self._info_label.set_text(" • ".join(parts))

    def _start_pulse(self) -> None:
        if self._pulse_source_id is None:
            self._progress_bar.set_pulse_step(0.08)
            self._pulse_source_id = GLib.timeout_add(110, self._on_pulse_tick)

    def _stop_pulse(self) -> None:
        if self._pulse_source_id is not None:
            GLib.source_remove(self._pulse_source_id)
            self._pulse_source_id = None

    def _on_pulse_tick(self) -> bool:
        self._progress_bar.pulse()
        return GLib.SOURCE_CONTINUE

    def _start_elapsed_timer(self) -> None:
        if self._timer_source_id is None:
            self._timer_source_id = GLib.timeout_add(500, self._on_elapsed_tick)

    def _stop_elapsed_timer(self) -> None:
        if self._timer_source_id is not None:
            GLib.source_remove(self._timer_source_id)
            self._timer_source_id = None

    def _on_elapsed_tick(self) -> bool:
        if self._last_state.get("status") not in ("loading", "transcribing", "pro"):
            self._timer_source_id = None
            return GLib.SOURCE_REMOVE
        self._update_info_label(self._last_state)
        return GLib.SOURCE_CONTINUE

    def _set_placeholder(self, text: str) -> None:
        self._text_view.add_css_class("placeholder-text")
        self._text_buffer.set_text(text)

    def _current_transcript_text(self) -> str:
        if not self._last_state:
            return ""
        use_ts = self._show_ts_check.get_active()
        arr = (
            self._last_state.get("ts_lines")
            if use_ts and self._last_state.get("ts_lines")
            else self._last_state.get("lines")
        )
        return "\n\n".join(arr) if arr else ""

    def _render_transcript(self, auto_scroll: bool = False) -> None:
        text = self._current_transcript_text()
        if not text:
            return
        vadj = self._scroll.get_vadjustment()
        at_bottom = (vadj.get_value() + vadj.get_page_size()) >= (
            vadj.get_upper() - 36.0
        )

        self._text_view.remove_css_class("placeholder-text")
        self._text_buffer.set_text(text)

        if auto_scroll and at_bottom:
            GLib.idle_add(self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> bool:
        vadj = self._scroll.get_vadjustment()
        vadj.set_value(max(0.0, vadj.get_upper() - vadj.get_page_size()))
        return GLib.SOURCE_REMOVE

    def _on_copy_clicked(self, _button: Gtk.Button) -> None:
        text = self._current_transcript_text()
        if not text:
            return
        display = Gdk.Display.get_default()
        if display is not None:
            display.get_clipboard().set(text)
        self._copy_btn.set_label("Copiato!")
        self._show_toast("Testo copiato negli appunti")
        if self._copy_reset_source_id is not None:
            GLib.source_remove(self._copy_reset_source_id)
        self._copy_reset_source_id = GLib.timeout_add(1500, self._reset_copy_label)

    def _reset_copy_label(self) -> bool:
        self._copy_btn.set_label("Copia testo")
        self._copy_reset_source_id = None
        return GLib.SOURCE_REMOVE

    def _launch_path(self, path_str: str) -> None:
        if not path_str:
            return
        target = Path(path_str)
        if not target.exists():
            self._show_toast("File o cartella non trovato.")
            return
        gfile = Gio.File.new_for_path(str(target))
        launcher = Gtk.FileLauncher.new(gfile)
        launcher.launch(self, None, self._on_launch_done)

    def _on_launch_done(
        self, launcher: Gtk.FileLauncher, result: Gio.AsyncResult
    ) -> None:
        try:
            launcher.launch_finish(result)
        except GLib.Error as err:
            gfile = launcher.get_file()
            if gfile is not None:
                try:
                    Gio.AppInfo.launch_default_for_uri(gfile.get_uri(), None)
                    return
                except GLib.Error:
                    pass
            self._show_toast(f"Impossibile aprire: {err.message}")

    def _on_open_txt_clicked(self, _button: Gtk.Button) -> None:
        self._launch_path(self._last_state.get("txt_path", ""))

    def _on_rename_voices_clicked(self, _button: Gtk.Button) -> None:
        speakers = self._last_state.get("speakers") or []
        if not speakers:
            return
        speaker_names = dict(self._last_state.get("speaker_names") or {})

        dialog = Adw.AlertDialog(
            heading="Rinomina voci",
            body="Assegna un nome a ciascuna voce rilevata. La trascrizione e il file .txt verranno aggiornati subito.",
        )
        dialog.add_response("cancel", "Annulla")
        dialog.add_response("apply", "Applica")
        dialog.set_response_appearance("apply", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("apply")
        dialog.set_close_response("cancel")

        form_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        form_box.set_margin_top(6)

        entries: dict[str, Gtk.Entry] = {}
        for spk_id in speakers:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            lbl = Gtk.Label(label=f"[{spk_id}]", xalign=0.0)
            lbl.set_size_request(90, -1)
            lbl.add_css_class("field-label")

            entry = Gtk.Entry()
            entry.set_hexpand(True)
            entry.set_text(speaker_names.get(spk_id, spk_id))
            entry.set_placeholder_text(spk_id)
            entry.connect(
                "activate",
                lambda _e, d=dialog: d.response("apply") or d.close(),
            )
            entries[spk_id] = entry

            row.append(lbl)
            row.append(entry)
            form_box.append(row)

        dialog.set_extra_child(form_box)
        dialog.connect("response", self._on_rename_dialog_response, entries)
        dialog.present(self)

    def _on_rename_dialog_response(
        self,
        _dialog: Adw.AlertDialog,
        response: str,
        entries: dict[str, Gtk.Entry],
    ) -> None:
        if response != "apply":
            return
        new_names = {
            spk_id: (entry.get_text().strip() or spk_id)
            for spk_id, entry in entries.items()
        }
        self._apply_speaker_renames(new_names)

    def _apply_speaker_renames(self, new_names: dict[str, str]) -> None:
        blocks = self._last_state.get("speaker_blocks") or []
        txt_path_str = self._last_state.get("txt_path", "")
        if not blocks and not txt_path_str:
            return

        current_names = dict(self._last_state.get("speaker_names") or {})
        if txt_path_str:
            txt_path = Path(txt_path_str)
            try:
                if blocks:
                    engine.save_blocks_to_txt(
                        txt_path, blocks, speaker_names=new_names
                    )
                else:
                    content = txt_path.read_text(encoding="utf-8")
                    txt_path.write_text(
                        engine.rename_speaker_tags(content, current_names, new_names),
                        encoding="utf-8",
                    )
            except OSError as err:
                self._show_toast(f"Errore salvataggio .txt: {err}")
                return

        if blocks:
            lines = engine.format_blocks(
                blocks, speaker_names=new_names, show_timestamps=False
            )
            ts_lines = engine.format_blocks(
                blocks, speaker_names=new_names, show_timestamps=True
            )
        else:
            lines = [
                engine.rename_speaker_tags(line, current_names, new_names)
                for line in self._last_state.get("lines", [])
            ]
            ts_lines = [
                engine.rename_speaker_tags(line, current_names, new_names)
                for line in self._last_state.get("ts_lines", [])
            ]
        self._last_state["speaker_names"] = dict(new_names)
        self._last_state["lines"] = lines
        self._last_state["ts_lines"] = ts_lines
        self._render_transcript(auto_scroll=False)
        if txt_path_str:
            self._show_toast(f"Voci rinominate e salvate in {Path(txt_path_str).name}")

    def _on_live_clicked(self, _button: Gtk.Button) -> None:
        if self._worker.is_busy():
            self._show_toast("Attendi la fine della trascrizione in corso o annullala.")
            return
        if self._live_transcriber is None:
            import live

            self._live_transcriber = live.LiveTranscriber(self._on_worker_update)

        if self._live_transcriber.is_running():
            self._live_btn.set_sensitive(False)
            self._live_btn.set_label("Salvataggio LIVE...")
            self._live_transcriber.stop()
        else:
            self._live_btn.set_label("Stop LIVE")
            self._live_btn.remove_css_class("secondary-btn")
            self._live_btn.add_css_class("destructive-action")
            self._live_transcriber.start(
                language=self._selected_language(),
                initial_prompt=self._current_prompt_text(),
            )

    def _on_open_dir_clicked(self, _button: Gtk.Button) -> None:
        engine.OUTPUT_DIR.mkdir(exist_ok=True)
        self._launch_path(str(engine.OUTPUT_DIR))

    def _on_cancel_clicked(self, _button: Gtk.Button) -> None:
        self._cancel_btn.set_sensitive(False)
        if self._live_transcriber is not None and self._live_transcriber.is_running():
            self._live_transcriber.stop()
        self._worker.cancel()

    def _show_toast(self, message: str) -> None:
        toast = Adw.Toast.new(message)
        toast.set_timeout(3)
        self._toast_overlay.add_toast(toast)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        if self._save_prompt_source_id is not None:
            GLib.source_remove(self._save_prompt_source_id)
            self._save_prompt_source_id = None
            self._persist_current_config()

        if self._pro_cleanup_done:
            return False
        if self._pro_cleanup_pending:
            return True

        live_busy = (
            self._live_transcriber is not None and self._live_transcriber.is_running()
        )
        if self._worker.is_busy() or live_busy:
            if self._closing_confirmed:
                self._stop_pulse()
                self._stop_elapsed_timer()
                if self._live_transcriber is not None:
                    self._live_transcriber.stop()
                self._worker.shutdown(timeout=0.1)
                self._pro_cleanup_pending = True
                GLib.timeout_add(250, self._wait_for_worker_before_close)
                return True

            dialog = Adw.AlertDialog(
                heading="Trascrizione in corso",
                body="Vuoi annullare la trascrizione attiva, liberare la GPU e chiudere l'applicazione?",
            )
            dialog.add_response("stay", "Continua")
            dialog.add_response("close", "Annulla e chiudi")
            dialog.set_response_appearance("close", Adw.ResponseAppearance.DESTRUCTIVE)
            dialog.set_default_response("stay")
            dialog.set_close_response("stay")
            dialog.connect("response", self._on_close_dialog_response)
            dialog.present(self)
            return True

        self._stop_pulse()
        self._stop_elapsed_timer()
        if self._live_transcriber is not None:
            self._live_transcriber.stop()
        self._worker.shutdown(timeout=0.1)
        self._start_pro_close_cleanup()
        return True

    def _wait_for_worker_before_close(self) -> bool:
        live_busy = (
            self._live_transcriber is not None and self._live_transcriber.is_running()
        )
        if self._worker.is_busy() or live_busy:
            self._worker.shutdown(timeout=0.1)
            return GLib.SOURCE_CONTINUE
        self._pro_cleanup_pending = False
        self._start_pro_close_cleanup()
        return GLib.SOURCE_REMOVE

    def _start_pro_close_cleanup(self) -> None:
        self._pro_cleanup_pending = True
        threading.Thread(
            target=self._release_pro_resources_before_close, daemon=True
        ).start()

    def _release_pro_resources_before_close(self) -> None:
        error = ""
        try:
            import pro

            pro.release_local_model()
        except Exception as exc:
            error = str(exc)
        GLib.idle_add(self._finish_pro_close_cleanup, error)

    def _finish_pro_close_cleanup(self, error: str) -> bool:
        self._pro_cleanup_pending = False
        if error:
            dialog = Adw.MessageDialog.new(
                self,
                "Impossibile liberare il modello PRO",
                f"{error}\nPuoi riprovare o chiudere comunque.",
            )
            dialog.add_response("retry", "Riprova")
            dialog.add_response("close", "Chiudi comunque")
            dialog.set_default_response("retry")
            dialog.set_close_response("close")
            dialog.connect("response", self._on_pro_close_cleanup_error)
            dialog.present()
        else:
            self._pro_cleanup_done = True
            self.close()
        return GLib.SOURCE_REMOVE

    def _on_pro_close_cleanup_error(
        self, _dialog: Adw.MessageDialog, response: str
    ) -> None:
        if response == "retry":
            self._start_pro_close_cleanup()
        else:
            self._pro_cleanup_done = True
            self.close()

    def _on_close_dialog_response(
        self, _dialog: Adw.AlertDialog, response: str
    ) -> None:
        if response == "close":
            self._closing_confirmed = True
            self._stop_pulse()
            self._stop_elapsed_timer()
            if self._live_transcriber is not None:
                self._live_transcriber.stop()
            self._worker.cancel()
            self._worker.shutdown(timeout=5.0)
            self.close()


class TranscriptionApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.HANDLES_OPEN,
        )
        self._window: TranscriptionWindow | None = None
        self._preload_scheduled = False

    def do_startup(self) -> None:
        Adw.Application.do_startup(self)
        GLib.set_application_name(APP_TITLE)
        GLib.set_prgname(APP_ID)

        close_action = Gio.SimpleAction.new("close", None)
        close_action.connect("activate", self._on_close_action)
        self.add_action(close_action)
        self.set_accels_for_action("app.close", ["<Primary>w", "<Primary>q"])

        style_manager = Adw.StyleManager.get_default()
        style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)

        display = Gdk.Display.get_default()
        if display is not None:
            provider = Gtk.CssProvider()
            provider.load_from_string(APP_CSS)
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def _on_close_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        if self._window is not None:
            self._window.close()

    def _ensure_window(self) -> TranscriptionWindow:
        if self._window is None:
            self._window = TranscriptionWindow(self)
        self._window.present()
        if not self._preload_scheduled:
            self._preload_scheduled = True
            GLib.idle_add(self._start_background_preload)
        return self._window

    def _start_background_preload(self) -> bool:
        def on_ready() -> None:
            GLib.idle_add(self._notify_backend_ready)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._notify_backend_error, str(exc))

        engine.preload_backend_async(on_ready=on_ready, on_error=on_error)
        return GLib.SOURCE_REMOVE

    def _notify_backend_ready(self) -> bool:
        if self._window is not None:
            self._window.set_backend_status("CUDA pronto", ready=True)
        return GLib.SOURCE_REMOVE

    def _notify_backend_error(self, message: str) -> bool:
        if self._window is not None:
            self._window.set_backend_status("Errore CUDA", ready=False)
            self._window._show_toast(f"Errore caricamento backend: {message}")
        return GLib.SOURCE_REMOVE

    def do_activate(self) -> None:
        self._ensure_window()

    def do_open(self, files: list[Gio.File], _n_files: int, _hint: str) -> None:
        win = self._ensure_window()
        paths = [Path(f.get_path()) for f in files if f.get_path()]
        if paths:
            win.enqueue_paths(paths)


def main() -> int:
    app = TranscriptionApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
