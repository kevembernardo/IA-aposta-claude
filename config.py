"""
Configurações do Betting AI Engine
====================================
Todas as variáveis vêm de variáveis de ambiente (.env)
para manter credenciais seguras no GitHub.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── CREDENCIAIS ──────────────────────────────────────────────
    anthropic_api_key: str  = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    telegram_token: str     = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    telegram_chat_id: str   = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # ── BANCA E GESTÃO ────────────────────────────────────────────
    banca: float            = field(default_factory=lambda: float(os.getenv("BANCA", "1000")))
    stop_loss_pct: float    = field(default_factory=lambda: float(os.getenv("STOP_LOSS_PCT", "5")))
    kelly_fraction: float   = field(default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25")))
    min_ev_pct: float       = field(default_factory=lambda: float(os.getenv("MIN_EV_PCT", "4")))

    # ── OPERAÇÃO ──────────────────────────────────────────────────
    scan_interval_min: int  = field(default_factory=lambda: int(os.getenv("SCAN_INTERVAL_MIN", "180")))  # 3h
    max_matches: int        = field(default_factory=lambda: int(os.getenv("MAX_MATCHES", "10")))
    summary_hour: int       = field(default_factory=lambda: int(os.getenv("SUMMARY_HOUR", "8")))  # 8h da manhã

    # ── LIGAS ─────────────────────────────────────────────────────
    leagues: List[str]      = field(default_factory=lambda: os.getenv(
        "LEAGUES",
        "Premier League,La Liga,Brasileirão Série A,Champions League,Serie A"
    ).split(","))

    def validate(self):
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY não definida")
        if not self.telegram_token:
            errors.append("TELEGRAM_TOKEN não definida")
        if not self.telegram_chat_id:
            errors.append("TELEGRAM_CHAT_ID não definida")
        if errors:
            raise ValueError("Configuração inválida:\n" + "\n".join(f"  • {e}" for e in errors))
        return True
