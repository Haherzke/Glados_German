# ruff: noqa: RUF001
import subprocess
from typing import Any

class Phonemizer:
    """
    Vereinfachter Phonemizer für deutsches GLaDOS TTS mit espeak-ng.
    
    Diese Klasse konvertiert Text direkt zu deutschen IPA-Phonemen mit espeak-ng,
    ohne komplexe ONNX-Modelle oder Dictionary-Lookups.
    """

    def __init__(self, config: Any = None) -> None:
        """
        Initialisiert den Phonemizer nur mit notwendigen Konfigurationen.
        
        Args:
            config: Wird für Kompatibilität akzeptiert, aber nicht verwendet
        """
        # Spezielle Phoneme für bekannte Begriffe
        self.phoneme_dict: dict[str, str] = {
            "glados": "ɡlˈɑːdɑːs",  # GLaDOS Spezialbehandlung
        }
        
        # # Deutsche Phonem-Verbesserungen für bessere Aussprache
        self.german_phoneme_fixes: dict[str, str] = {
            # CH-Laut Verbesserungen
            "ç": "ç",  # ich-Laut (bleibt gleich, aber explizit)
            "x": "x",  # ach-Laut (bleibt gleich, aber explizit)
        #     # Umlaute-Verbesserungen
        #     "ɛː": "ɛː",  # ä (lang)
        #     "øː": "øː",  # ö (lang)  
        #     "yː": "yː",  # ü (lang)
        #     "ɛ": "ɛ",    # ä (kurz)
        #     "ø": "ø",    # ö (kurz)
        #     "y": "y",    # ü (kurz)
        #     # Konsonanten-Verbesserungen
        #     "ʃ": "ʃ",    # sch-Laut
        #     "ʒ": "ʒ",    # stimmhafter sch-Laut (Genre)
        #     "pf": "pf",  # pf-Laut
        #     "ts": "ts",  # z-Laut
         }
        
        # # Spezielle deutsche Wort-zu-Phonem Mappings für schwierige Wörter
        self.german_word_fixes: dict[str, str] = {
            # Häufige CH-Wörter mit korrekter Aussprache
        #    "ich": "ˈɪç",
        #     "nicht": "nˈɪçt", 
        #     "machen": "mˈaxən",
        #     "nacht": "nˈaxt",
        #     "licht": "lˈɪçt",
        #     "küche": "kˈyçə",
        #     "tochter": "tˈɔxtɐ",
        #     "sprechen": "ʃpʁˈɛçən",
        #     "rechnen": "ʁˈɛçnən",
        #     "lachen": "lˈaxən",
        #     # Weitere schwierige deutsche Wörter
        #     "träume": "tʁˈɔʏmə",
        #     "schön": "ʃøːn",
        #     "größer": "ɡʁøːsɐ",
        #     "müssen": "mˈʏsən",
        #     "können": "kˈønən",
        #     "hören": "høːʁən",
         }
        
        # # Prüfe ob espeak-ng verfügbar ist
        # self._check_espeak_ng()

    def _check_espeak_ng(self) -> bool:
        """
        Prüft ob espeak-ng verfügbar ist.
        
        Returns:
            bool: True wenn espeak-ng verfügbar ist
            
        Raises:
            RuntimeError: Wenn espeak-ng nicht gefunden wird
        """
        try:
            subprocess.run(['espeak-ng', '--version'], 
                         capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "espeak-ng nicht gefunden. Bitte installieren Sie espeak-ng:\n"
                "Linux: sudo apt install espeak-ng\n"
                "Windows: choco install espeak-ng"
            )

    def convert_to_phonemes(self, texts: list[str], lang: str = "de") -> list[str]:
        """
        Konvertiert eine Liste von Texten zu deutschen IPA-Phonemen mit espeak-ng.
        
        Diese Methode ersetzt die komplexe ONNX-basierte Phonemisierung durch
        direkte espeak-ng Aufrufe, die native deutsche Phonemisierung bieten.
        Verwendet verbesserte wortweise Phonemisierung für bessere deutsche Aussprache.
        
        Args:
            texts (list[str]): Liste von Texten zur Konvertierung
            lang (str): Sprache für espeak-ng (Standard: "de")
            
        Returns:
            list[str]: Liste der IPA-Phonem-Strings
            
        Example:
            phonemizer = Phonemizer()
            phonemes = phonemizer.convert_to_phonemes(["Hallo Welt"])
            # Gibt etwa zurück: ["halˈoː vˈɛlt"]
        """

        # results = []
        # for text in texts:
        #     # Einfache espeak-ng Phonemisierung ohne zu viele Fixes
        #     try:
        #         result = subprocess.run(
        #             ['espeak-ng', '-q', '--ipa=3', '-v', 'de', text],
        #             capture_output=True, text=True, check=True
        #         )
        #         phonemes = result.stdout.strip()
        #         results.append(phonemes)
        #     except Exception:
        #         results.append(text.lower())
        # return results

        phoneme_results = []
        
        for text in texts:
            if not text or not text.strip():
                phoneme_results.append("")
                continue
            
            # Verbesserte wortweise Phonemisierung
            phonemes = self._phonemize_with_word_fixes(text, lang)
            phoneme_results.append(phonemes)
        
        print(f"Phoneme für die Eingabetexte: {phoneme_results}")
        return phoneme_results



    def _phonemize_with_word_fixes(self, text: str, lang: str = "de") -> str:
        """
        Phonemisiert Text wortweise mit speziellen deutschen Korrekturen.
        
        Args:
            text (str): Text zur Phonemisierung
            lang (str): Sprache für espeak-ng
            
        Returns:
            str: IPA-Phoneme mit Verbesserungen
        """
        import re
        
        # Teile Text in Wörter und Interpunktion
        words = re.findall(r'\b\w+\b|\W+', text)
        phoneme_parts = []
        
        for word in words:
            word_lower = word.lower().strip()
            
            # Prüfe auf spezielle deutsche Wort-Fixes
            if word_lower in self.german_word_fixes:
                phoneme_parts.append(self.german_word_fixes[word_lower])
            elif word_lower in self.phoneme_dict:
                phoneme_parts.append(self.phoneme_dict[word_lower])
            elif word.strip() and word.strip().isalpha():
                # Normales Wort mit espeak-ng phonemisieren
                try:
                    result = subprocess.run(
                        ['espeak-ng', '-q', '--ipa=2', '-v', lang, word],
                        capture_output=True, text=True, check=True, encoding='utf-8'
                    )
                    phonemes = result.stdout.strip() if result.stdout else word.lower()
                    # Deutsche Phonem-Verbesserungen anwenden
                    phonemes = self._apply_german_phoneme_fixes(phonemes)
                    phoneme_parts.append(phonemes)
                except Exception as e:
                    print(f"espeak-ng Fehler für Wort '{word}': {e}")
                    phoneme_parts.append(word.lower())
            else:
                # Interpunktion oder Zahlen - unverändert lassen
                phoneme_parts.append(word)
        
        return "".join(phoneme_parts)

    def _preprocess_text(self, text: str) -> str:
        """
        Vorverarbeitung des Textes vor der Phonemisierung.
        
        Args:
            text (str): Originaltext
            
        Returns:
            str: Vorverarbeiteter Text
        """
        processed = text.lower()  # Normalisiere zu Kleinschreibung für Vergleiche
        
        # GLaDOS Spezialbehandlung - ersetze vor espeak-ng Aufruf
        for word, phoneme in self.phoneme_dict.items():
            if word.lower() in processed:
                # Temporär ersetzen um espeak-ng Konfusion zu vermeiden
                processed = processed.replace(word.lower(), f"_{word}_")
        
        # Deutsche Wort-Fixes - markiere spezielle Wörter
        for word, _ in self.german_word_fixes.items():
            if word in processed:
                # Markiere Wörter für spezielle Behandlung
                processed = processed.replace(word, f"<{word}>")
        
        return processed

    def _postprocess_phonemes(self, phonemes: str, original_text: str) -> str:
        """
        Nachbearbeitung der Phoneme für spezielle Begriffe und deutsche Besonderheiten.
        
        Args:
            phonemes (str): Von espeak-ng generierte Phoneme
            original_text (str): Originaltext für Kontext
            
        Returns:
            str: Nachbearbeitete Phoneme
        """
        processed = phonemes
        
        # GLaDOS Spezialbehandlung - ersetze in Phonemen
        for word, special_phoneme in self.phoneme_dict.items():
            if word.lower() in original_text.lower():
                # Ersetze die Placeholder-Phoneme mit den speziellen
                placeholder_pattern = f"_{word}_"
                if placeholder_pattern in original_text:
                    processed = processed.replace(phonemes, special_phoneme)
        
        # Deutsche Wort-Fixes anwenden
        processed = self._apply_german_word_fixes(processed, original_text)
        
        # Deutsche Phonem-Verbesserungen anwenden
        processed = self._apply_german_phoneme_fixes(processed)
        
        return processed

    def _apply_german_word_fixes(self, phonemes: str, original_text: str) -> str:
        """
        Wendet spezielle deutsche Wort-zu-Phonem Korrekturen an.
        
        Args:
            phonemes (str): Ursprüngliche Phoneme
            original_text (str): Originaltext für Wort-Matching
            
        Returns:
            str: Korrigierte Phoneme
        """
        processed = phonemes
        original_lower = original_text.lower()
        
        # Durchgehe alle deutschen Wort-Fixes
        for word, correct_phoneme in self.german_word_fixes.items():
            # Prüfe ob das Wort im Original vorkommt (als ganzes Wort)
            import re
            word_pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(word_pattern, original_lower):
                # Ersetze die von espeak generierten Phoneme für dieses Wort
                # Dies ist vereinfacht - in einer vollständigen Lösung würde man
                # die Phoneme wortweise zuordnen
                marker = f"<{word}>"
                if marker in original_text:
                    # Hier könnten wir präziser die entsprechenden Phoneme ersetzen
                    pass
        
        return processed

    def _apply_german_phoneme_fixes(self, phonemes: str) -> str:
        """
        Wendet deutsche Phonem-Verbesserungen an für bessere Aussprache.
        
        Args:
            phonemes (str): Ursprüngliche Phoneme
            
        Returns:
            str: Verbesserte Phoneme
        """
        processed = phonemes
        
        # Spezielle deutsche CH-Laut Verbesserungen
        # Ersetze problematische Kombinationen
        processed = processed.replace("kh", "x")  # Falls espeak kh statt x generiert
        processed = processed.replace("χ", "x")   # Falls chi statt x verwendet wird
        
        # Verbessere R-Laute (deutsches R oft problematisch)
        processed = processed.replace("ʁ", "ʁ")   # Uvulares R beibehalten
        processed = processed.replace("r̥", "ʁ")   # Stimmloses R durch uvulares ersetzen
        
        # Verbessere lange Vokale (oft zu kurz ausgesprochen)
        processed = processed.replace("aː", "aː")  # Langes A beibehalten
        processed = processed.replace("eː", "eː")  # Langes E beibehalten
        processed = processed.replace("iː", "iː")  # Langes I beibehalten
        processed = processed.replace("oː", "oː")  # Langes O beibehalten
        processed = processed.replace("uː", "uː")  # Langes U beibehalten
        
        # Spezielle deutsche Konsonanten-Cluster
        processed = processed.replace("kv", "kv")  # qu-Laut
        processed = processed.replace("ʔ", "")     # Glottaler Stopp oft problematisch, entfernen
        
        return processed

    # Kompatibilitätsmethoden für bestehenden Code
    
    def phonemize(self, text: str, lang: str = "de") -> str:
        """
        Kompatibilitätsmethode für einzelnen Text.
        
        Args:
            text (str): Text zur Phonemisierung
            lang (str): Sprache
            
        Returns:
            str: IPA-Phoneme
        """
        result = self.convert_to_phonemes([text], lang)
        return result[0] if result else ""

    def __del__(self) -> None:
        """
        Cleanup - hier nicht nötig da keine ONNX Sessions
        """
        pass


# Fallback-Konfigurationsklasse für Kompatibilität
class ModelConfig:
    """
    Dummy-Konfiguration für Kompatibilität mit bestehendem Code.
    Da wir espeak-ng direkt verwenden, werden keine Modell-Pfade benötigt.
    """
    
    def __init__(self, **kwargs):
        """Akzeptiert alle Parameter für Kompatibilität, verwendet sie aber nicht."""
        pass


# Spezielle Token-Enums für Kompatibilität (falls im Hauptcode referenziert)
class SpecialTokens:
    PAD = "_"
    START = "<start>"
    END = "<end>"
    EN_US = "<en_us>"
    DE = "<de>"


class Punctuation:
    PUNCTUATION = "().,:?!/–"
    HYPHEN = "-"
    SPACE = " "