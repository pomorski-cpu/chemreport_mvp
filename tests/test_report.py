from core.report import _resolve_pdf_fonts, render_report_html


class _FakePdfMetrics:
    def __init__(self):
        self._registered_names = set()
        self.registered_fonts = []

    def getRegisteredFontNames(self):
        return list(self._registered_names)

    def registerFont(self, font):
        self._registered_names.add(font.fontName)
        self.registered_fonts.append((font.fontName, font.path))


class _FakeTTFont:
    def __init__(self, fontName, path):
        self.fontName = fontName
        self.path = path


def test_resolve_pdf_fonts_uses_unicode_ttf_when_available(monkeypatch):
    fake_pdfmetrics = _FakePdfMetrics()
    existing = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    }

    monkeypatch.setattr("core.report.resource_path", lambda rel_path: f"/missing/{rel_path}")
    monkeypatch.setattr("core.report.os.path.exists", lambda path: path in existing)

    regular, bold = _resolve_pdf_fonts(fake_pdfmetrics, _FakeTTFont)

    assert (regular, bold) == ("ChemReportUnicode", "ChemReportUnicode-Bold")
    assert fake_pdfmetrics.registered_fonts == [
        ("ChemReportUnicode", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("ChemReportUnicode-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]


def test_resolve_pdf_fonts_falls_back_to_helvetica_when_no_ttf_found(monkeypatch):
    fake_pdfmetrics = _FakePdfMetrics()

    monkeypatch.setattr("core.report.resource_path", lambda rel_path: f"/missing/{rel_path}")
    monkeypatch.setattr("core.report.os.path.exists", lambda path: False)

    regular, bold = _resolve_pdf_fonts(fake_pdfmetrics, _FakeTTFont)

    assert (regular, bold) == ("Helvetica", "Helvetica-Bold")
    assert fake_pdfmetrics.registered_fonts == []


def test_render_report_html_includes_analogues_for_each_read_across_target():
    html = render_report_html(
        {
            "generated_at": "2026-03-10T10:00:00",
            "meta": {"smiles": "CCO"},
            "descriptors": {},
            "predictions": [],
            "warnings": [],
            "decision": {},
            "profile": {},
            "analogues": [],
            "category": {"summary_ru": "Сводка по аналогам."},
            "reliability": {},
            "svg": "",
            "read_across": {
                "targets": {
                    "toxicity": {
                        "label_ru": "Генотоксичность",
                        "prediction": {"value": "Генотоксичный", "confidence": "Высокая"},
                        "analogues": [
                            {
                                "rank": 1,
                                "similarity": 0.91,
                                "value": "Генотоксичный",
                                "class_name": "Гербицид",
                                "smiles": "CCN",
                            }
                        ],
                    },
                    "pesticide_class": {
                        "label_ru": "Класс пестицида",
                        "prediction": {"value": "Гербицид", "confidence": "Средняя"},
                        "analogues": [
                            {
                                "rank": 1,
                                "similarity": 0.88,
                                "value": "Гербицид",
                                "class_name": "Гербицид",
                                "smiles": "CCCl",
                            }
                        ],
                    },
                }
            },
        }
    )

    assert "Генотоксичность" in html
    assert "Класс пестицида" in html
    assert "CCN" in html
    assert "CCCl" in html
