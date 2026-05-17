"""
Motor Matemático — Poisson, Kelly Criterion, Expected Value
============================================================
Núcleo quantitativo do sistema. Puro Python, sem dependências externas.
"""

import math
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MarketAnalysis:
    key: str
    label: str
    real_prob: float
    implied_prob: float
    odds: float
    ev: float
    ev_pct: float
    edge_pct: float
    kelly_full: float
    kelly_frac: float
    stake: float
    cls_label: str
    cls_color: str  # para o Telegram (emoji)


def poisson_prob(lam: float, k: int) -> float:
    """P(X=k) usando log para estabilidade numérica."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = -lam + k * math.log(lam)
    for i in range(1, k + 1):
        log_p -= math.log(i)
    return math.exp(log_p)


def match_probabilities(lambda_home: float, lambda_away: float, max_goals: int = 8) -> Dict[str, float]:
    """
    Calcula probabilidades de todos os mercados via Distribuição de Poisson.
    λ_home = ataque_casa × defesa_fora
    λ_away = ataque_fora × defesa_casa
    """
    home_win = draw = away_win = over25 = btts = 0.0

    for h in range(max_goals + 1):
        ph = poisson_prob(lambda_home, h)
        for a in range(max_goals + 1):
            pa = poisson_prob(lambda_away, a)
            p = ph * pa
            if h > a:   home_win += p
            elif h == a: draw     += p
            else:        away_win += p
            if h + a > 2.5: over25 += p
            if h > 0 and a > 0: btts += p

    return {
        "home":    home_win,
        "draw":    draw,
        "away":    away_win,
        "over25":  over25,
        "under25": 1.0 - over25,
        "btts":    btts,
        "no_btts": 1.0 - btts,
    }


def expected_value(real_prob: float, odds: float) -> float:
    """EV = prob × (odds - 1) - (1 - prob)"""
    return real_prob * (odds - 1) - (1 - real_prob)


def implied_probability(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.0


def edge_percent(real_prob: float, odds: float) -> float:
    return (real_prob - implied_probability(odds)) * 100


def kelly_criterion(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Kelly fracionado. Retorna 0 se negativo (não apostar)."""
    b = odds - 1
    if b <= 0:
        return 0.0
    k = (b * prob - (1 - prob)) / b
    return max(0.0, k * fraction)


def classify_opportunity(ev: float, edge: float) -> tuple:
    """Retorna (label, emoji) baseado na força da oportunidade."""
    if ev > 0.12 and edge > 5:
        return "ALTO VALOR",     "🟢"
    if ev > 0.06 and edge > 2:
        return "BOM VALOR",      "🔵"
    if ev > 0.01 and edge > 0:
        return "VALOR MARGINAL", "🟡"
    if ev >= 0:
        return "SEM VANTAGEM",   "🟠"
    return "EVITAR", "🔴"


MARKET_LABELS = {
    "home":    "Vitória Casa",
    "draw":    "Empate",
    "away":    "Vitória Fora",
    "over25":  "Over 2.5 Gols",
    "under25": "Under 2.5 Gols",
    "btts":    "Ambas Marcam",
    "no_btts": "Ambas NÃO Marcam",
}


def analyze_match(match_data: dict, banca: float, kelly_fraction: float = 0.25) -> List[MarketAnalysis]:
    """
    Recebe dados de um jogo e retorna análise completa de todos os mercados.
    """
    lh = match_data["homeAvgGoalsScored"] * match_data["awayAvgGoalsConceded"]
    la = match_data["awayAvgGoalsScored"] * match_data["homeAvgGoalsConceded"]
    probs = match_probabilities(lh, la)

    odds_map = match_data.get("odds", {})
    results = []

    for key, real_prob in probs.items():
        odds = odds_map.get(key)
        if not odds or odds <= 1.01:
            continue

        ev      = expected_value(real_prob, odds)
        ev_pct  = ev * 100
        edge    = edge_percent(real_prob, odds)
        kf      = kelly_criterion(real_prob, odds, kelly_fraction)
        k_full  = kelly_criterion(real_prob, odds, 1.0)
        stake   = banca * kf
        imp_p   = implied_probability(odds)
        lbl, emoji = classify_opportunity(ev, edge)

        results.append(MarketAnalysis(
            key=key,
            label=MARKET_LABELS.get(key, key),
            real_prob=real_prob,
            implied_prob=imp_p,
            odds=odds,
            ev=ev,
            ev_pct=ev_pct,
            edge_pct=edge,
            kelly_full=k_full,
            kelly_frac=kf,
            stake=stake,
            cls_label=lbl,
            cls_color=emoji,
        ))

    return sorted(results, key=lambda x: x.ev_pct, reverse=True)
