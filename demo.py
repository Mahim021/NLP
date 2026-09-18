"""MailSense GUI demo: compare TF-IDF, LSTM and BERT predictions side by side."""

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.predict import LOADERS  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402

MODELS_DIR = ROOT / "models"
THRESHOLD = 0.5
MODEL_NAMES = {
    "tfidf": "TF-IDF + Logistic Regression",
    "lstm": "LSTM + Word2Vec",
    "bert": "BERT",
}
EXAMPLES = [
    "Please review the attached report and send your feedback by Friday.",
    "Congratulations, you won a free prize! Click here to claim it.",
    "Can you join the project meeting tomorrow at 10 AM?",
    "Our weekly newsletter: top stories from around the web.",
]

ACTIONABLE_COLOR = "#1b7f3b"
NON_ACTIONABLE_COLOR = "#b3261e"
MUTED_COLOR = "#666666"


class ModelCard(ttk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, padding=12, relief="groove", borderwidth=1)
        ttk.Label(self, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.verdict = tk.Label(self, text="Loading model...", font=("Segoe UI", 14, "bold"),
                                fg=MUTED_COLOR)
        self.verdict.pack(anchor="w", pady=(8, 4))
        self.bar = ttk.Progressbar(self, maximum=100, length=220)
        self.bar.pack(fill="x")
        self.confidence = ttk.Label(self, text="Confidence: -")
        self.confidence.pack(anchor="w", pady=(4, 0))

    def set_status(self, text):
        self.verdict.config(text=text, fg=MUTED_COLOR)
        self.bar["value"] = 0
        self.confidence.config(text="Confidence: -")

    def set_result(self, probability):
        actionable = probability >= THRESHOLD
        confidence = probability if actionable else 1 - probability
        self.verdict.config(
            text="Actionable" if actionable else "Non-Actionable",
            fg=ACTIONABLE_COLOR if actionable else NON_ACTIONABLE_COLOR,
        )
        self.bar["value"] = confidence * 100
        self.confidence.config(text=f"Confidence: {confidence:.1%}")


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.predictors = {}
        self.events = queue.Queue()

        root.title("MailSense Demo")
        root.geometry("820x560")
        root.minsize(700, 480)

        main = ttk.Frame(root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="MailSense: Actionable E-mail Detection",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(main, text="Type or paste an e-mail, then click Predict to compare all three models.",
                  foreground=MUTED_COLOR).pack(anchor="w", pady=(2, 10))

        self.text = tk.Text(main, height=8, wrap="word", font=("Segoe UI", 11))
        self.text.pack(fill="both", expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=10)
        self.predict_button = ttk.Button(controls, text="Predict", command=self.predict,
                                         state="disabled")
        self.predict_button.pack(side="left")
        ttk.Button(controls, text="Clear", command=self.clear).pack(side="left", padx=6)
        ttk.Label(controls, text="Example:").pack(side="left", padx=(16, 4))
        self.example = ttk.Combobox(controls, values=EXAMPLES, state="readonly", width=50)
        self.example.pack(side="left", fill="x", expand=True)
        self.example.bind("<<ComboboxSelected>>", self.use_example)

        cards = ttk.Frame(main)
        cards.pack(fill="x")
        self.cards = {}
        for column, (key, title) in enumerate(MODEL_NAMES.items()):
            card = ModelCard(cards, title)
            card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
            cards.columnconfigure(column, weight=1)
            self.cards[key] = card

        self.status = ttk.Label(main, text="Loading models...", foreground=MUTED_COLOR)
        self.status.pack(anchor="w", pady=(10, 0))

        self.text.bind("<Control-Return>", lambda _event: self.predict() or "break")
        threading.Thread(target=self.load_models, daemon=True).start()
        self.root.after(100, self.process_events)

    def load_models(self):
        for key in MODEL_NAMES:
            try:
                self.predictors[key] = LOADERS[key](MODELS_DIR)
                self.events.put(("loaded", key, None))
            except Exception as exc:  # show the failure on the card, keep the others usable
                self.events.put(("load_error", key, exc))
        self.events.put(("all_loaded", None, None))

    def run_predictions(self, text):
        cleaned = [clean_text(text)]
        for key, predictor in self.predictors.items():
            try:
                self.events.put(("result", key, predictor(cleaned)[0]))
            except Exception as exc:
                self.events.put(("predict_error", key, exc))
        self.events.put(("done", None, None))

    def process_events(self):
        while not self.events.empty():
            event, key, value = self.events.get()
            if event == "loaded":
                self.cards[key].set_status("Ready")
            elif event == "load_error":
                self.cards[key].set_status("Failed to load")
                print(f"[{key}] load error: {value}", file=sys.stderr)
            elif event == "all_loaded":
                self.status.config(text=f"{len(self.predictors)} of {len(MODEL_NAMES)} models ready.")
                if self.predictors:
                    self.predict_button.config(state="normal")
            elif event == "result":
                self.cards[key].set_result(value)
            elif event == "predict_error":
                self.cards[key].set_status("Prediction failed")
                print(f"[{key}] prediction error: {value}", file=sys.stderr)
            elif event == "done":
                self.status.config(text="Done.")
                self.predict_button.config(state="normal")
        self.root.after(100, self.process_events)

    def predict(self):
        text = self.text.get("1.0", "end").strip()
        if not text or not self.predictors:
            self.status.config(text="Please enter some e-mail text first." if not text else self.status["text"])
            return
        self.predict_button.config(state="disabled")
        self.status.config(text="Predicting...")
        for key in self.predictors:
            self.cards[key].set_status("Predicting...")
        threading.Thread(target=self.run_predictions, args=(text,), daemon=True).start()

    def clear(self):
        self.text.delete("1.0", "end")
        self.example.set("")
        for key in self.predictors:
            self.cards[key].set_status("Ready")

    def use_example(self, _event):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.example.get())


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
