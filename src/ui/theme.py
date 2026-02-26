# src/ui/theme.py

# ── Color Palette ───────────────────────────────────────────────────────────
DARK_BG        = "#0F1923"   # Deep navy-black — main background
SURFACE        = "#1A2535"   # Slightly lighter — card/panel surfaces
SURFACE_RAISED = "#223044"   # Hover/active surface
BORDER         = "#2D3F55"   # Subtle borders
BORDER_BRIGHT  = "#3A5068"   # Focused/active borders

ACCENT_BLUE    = "#2979FF"   # Primary interactive elements
ACCENT_BLUE_DK = "#1A56CC"   # Hover state
ACCENT_CYAN    = "#00B4D8"   # Secondary accent (metrics, highlights)

SUCCESS        = "#00C853"   # Good stability, positive feedback
WARNING        = "#FFB300"   # Medium values, caution
DANGER         = "#FF3D3D"   # Poor stability, errors

TEXT_PRIMARY   = "#E8EDF3"   # Main text
TEXT_SECONDARY = "#8FA3BC"   # Labels, captions
TEXT_MUTED     = "#4F6178"   # Disabled / hint text

# ── Typography ───────────────────────────────────────────────────────────────
FONT_FAMILY    = "Inter, SF Pro Display, Segoe UI, Arial"
FONT_SM        = "11px"
FONT_MD        = "13px"
FONT_LG        = "15px"
FONT_XL        = "20px"
FONT_2XL       = "26px"

# ── Shared Styles ────────────────────────────────────────────────────────────
def card_style(hover=False):
    bg = SURFACE_RAISED if hover else SURFACE
    return f"""
        background-color: {bg};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px;
    """

def button_primary():
    return f"""
        QPushButton {{
            background-color: {ACCENT_BLUE};
            color: white;
            border: none;
            padding: 9px 20px;
            border-radius: 6px;
            font-size: {FONT_MD};
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {ACCENT_BLUE_DK}; }}
        QPushButton:pressed {{ background-color: #1040A0; }}
        QPushButton:disabled {{ background-color: {TEXT_MUTED}; color: {BORDER}; }}
    """

def button_danger():
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {DANGER};
            border: 1px solid {DANGER};
            padding: 9px 20px;
            border-radius: 6px;
            font-size: {FONT_MD};
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: rgba(255,61,61,0.12); }}
    """

def button_outlined():
    """Added helper for outlined buttons used in Live Analysis (Record Shot)"""
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {ACCENT_CYAN};
            border: 1px solid {ACCENT_CYAN};
            padding: 9px 20px;
            border-radius: 6px;
            font-size: {FONT_MD};
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: rgba(0, 180, 216, 0.12); }}
        QPushButton:disabled {{ border-color: {TEXT_MUTED}; color: {TEXT_MUTED}; }}
    """

APP_STYLESHEET = f"""
    QMainWindow, QWidget {{
        background-color: {DARK_BG};
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
        font-size: {FONT_MD};
    }}
    QLabel {{ color: {TEXT_PRIMARY}; background: transparent; }}
    QScrollBar:vertical {{
        background: {SURFACE};
        width: 6px;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_BRIGHT};
        border-radius: 3px;
        min-height: 30px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background: {SURFACE};
        height: 6px;
        border-radius: 3px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_BRIGHT};
        border-radius: 3px;
    }}
    QToolTip {{
        background-color: {SURFACE_RAISED};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_BRIGHT};
        border-radius: 4px;
        padding: 4px 8px;
    }}
"""

SIDEBAR_STYLE = f"""
    QWidget {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
"""

NAV_BUTTON_STYLE = f"""
    QPushButton {{
        background: transparent;
        color: {TEXT_SECONDARY};
        border: none;
        border-left: 3px solid transparent;
        padding: 14px 20px;
        text-align: left;
        font-size: {FONT_MD};
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {SURFACE_RAISED};
        color: {TEXT_PRIMARY};
    }}
    QPushButton:checked {{
        background-color: {SURFACE_RAISED};
        color: white;
        border-left: 3px solid {ACCENT_BLUE};
        font-weight: 600;
    }}
"""

INPUT_STYLE = f"""
    QLineEdit {{
        background-color: {DARK_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 10px 14px;
        font-size: {FONT_MD};
    }}
    QLineEdit:focus {{
        border: 1px solid {ACCENT_BLUE};
        background-color: #0F1E35;
    }}
    QLineEdit::placeholder {{
        color: {TEXT_MUTED};
    }}
"""
