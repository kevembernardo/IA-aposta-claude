"""
Memory — Sistema de Aprendizado e Histórico
===========================================
Registra todas as apostas sugeridas, resultados (quando informados)
e adapta os limites de EV com base no desempenho real.
"""

import json
import logging
import os
from datetime import datetime, date
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class Memory:
    def __init__(self, path: str = "data/history.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Erro ao carregar histórico: {e}. Iniciando do zero.")
        return {
            "sent_tips": [],
            "league_stats": {},
            "market_stats": {},
            "daily_stats": {},
            "performance": {
                "total_sent": 0,
                "total_wins": 0,
                "total_losses": 0,
                "total_staked": 0.0,
                "total_returned": 0.0,
            }
        }

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Erro ao salvar histórico: {e}")

    # ── CONTROLE DE ENVIO ─────────────────────────────────────────────────────

    def already_sent_today(self, match_key: str, market_key: str) -> bool:
        """Verifica se esse jogo+mercado já foi enviado hoje."""
        today = date.today().isoformat()
        return any(
            t["match_key"] == match_key
            and t["market_key"] == market_key
            and t["sent_date"] == today
            for t in self.data["sent_tips"]
        )

    def record_sent(self, match_key: str, opportunity: dict):
        """Registra um sinal enviado."""
        record = {
            "id":           len(self.data["sent_tips"]) + 1,
            "match_key":    match_key,
            "market_key":   opportunity["market_key"],
            "market_label": opportunity["market_label"],
            "home_team":    opportunity["home_team"],
            "away_team":    opportunity["away_team"],
            "league":       opportunity["league"],
            "date":         opportunity["date"],
            "odds":         opportunity["odds"],
            "stake":        opportunity["stake"],
            "ev_pct":       opportunity["ev_pct"],
            "edge_pct":     opportunity["edge_pct"],
            "sent_date":    date.today().isoformat(),
            "sent_at":      datetime.now().isoformat(),
            "result":       None,   # será preenchido depois
            "outcome":      None,   # "win" | "loss" | "void"
            "returned":     None,
        }
        self.data["sent_tips"].append(record)
        self.data["performance"]["total_sent"] += 1

        # Manter apenas últimos 500 registros em memória
        if len(self.data["sent_tips"]) > 500:
            self.data["sent_tips"] = self.data["sent_tips"][-500:]

        self._save()
        log.info(f"Registrado: {match_key} — {opportunity['market_label']} EV {opportunity['ev_pct']:.1f}%")

    def record_result(self, tip_id: int, outcome: str, returned: float = 0.0):
        """
        Registra o resultado de um sinal.
        outcome: "win" | "loss" | "void"
        returned: valor retornado pela aposta
        """
        for tip in self.data["sent_tips"]:
            if tip["id"] == tip_id:
                tip["outcome"] = outcome
                tip["returned"] = returned
                tip["result_at"] = datetime.now().isoformat()

                perf = self.data["performance"]
                perf["total_staked"] += tip["stake"]
                if outcome == "win":
                    perf["total_wins"]    += 1
                    perf["total_returned"] += returned
                elif outcome == "loss":
                    perf["total_losses"]   += 1

                # Atualizar stats por liga e mercado
                self._update_stats(tip["league"], tip["market_key"], outcome, tip["stake"], returned)
                self._save()
                log.info(f"Resultado registrado: tip #{tip_id} — {outcome}")
                return True
        return False

    def _update_stats(self, league: str, market: str, outcome: str, stake: float, returned: float):
        for stats_dict, key in [(self.data["league_stats"], league), (self.data["market_stats"], market)]:
            if key not in stats_dict:
                stats_dict[key] = {"sent": 0, "wins": 0, "losses": 0, "staked": 0.0, "returned": 0.0}
            s = stats_dict[key]
            s["sent"]    += 1
            s["staked"]  += stake
            s["returned"] += returned
            if outcome == "win":   s["wins"]   += 1
            if outcome == "loss":  s["losses"] += 1

    # ── APRENDIZADO ADAPTATIVO ────────────────────────────────────────────────

    def get_adaptive_min_ev(self, base_min_ev: float) -> float:
        """
        Ajusta o EV mínimo baseado no histórico recente.
        Se win rate > 55% → pode relaxar um pouco o limiar.
        Se win rate < 40% → aumenta o limiar (mais conservador).
        """
        perf = self.data["performance"]
        total = perf["total_wins"] + perf["total_losses"]

        if total < 20:
            # Dados insuficientes, usa base
            return base_min_ev

        win_rate = perf["total_wins"] / total

        if win_rate > 0.58:
            adjusted = max(base_min_ev * 0.85, 2.0)
            log.info(f"Win rate alto ({win_rate:.1%}) → EV mínimo relaxado para {adjusted:.1f}%")
            return adjusted
        elif win_rate < 0.38:
            adjusted = base_min_ev * 1.25
            log.info(f"Win rate baixo ({win_rate:.1%}) → EV mínimo aumentado para {adjusted:.1f}%")
            return adjusted

        return base_min_ev

    def get_league_performance(self, league: str) -> Optional[dict]:
        s = self.data["league_stats"].get(league)
        if not s or s["sent"] < 5:
            return None
        total = s["wins"] + s["losses"]
        return {
            "win_rate": s["wins"] / total if total > 0 else 0,
            "roi": (s["returned"] - s["staked"]) / s["staked"] * 100 if s["staked"] > 0 else 0,
            "sample": s["sent"],
        }

    # ── STATS DIÁRIAS ─────────────────────────────────────────────────────────

    def get_daily_stats(self) -> dict:
        today = date.today().isoformat()
        today_tips = [t for t in self.data["sent_tips"] if t["sent_date"] == today]
        wins   = sum(1 for t in today_tips if t["outcome"] == "win")
        losses = sum(1 for t in today_tips if t["outcome"] == "loss")
        best_ev = max((t["ev_pct"] for t in today_tips), default=0.0)
        staked   = sum(t["stake"]    for t in today_tips if t["outcome"] in ("win","loss"))
        returned = sum(t.get("returned", 0) for t in today_tips if t["outcome"] == "win")
        roi = (returned - staked) / staked * 100 if staked > 0 else 0.0

        return {
            "sent_today":  len(today_tips),
            "wins":        wins,
            "losses":      losses,
            "roi_today":   roi,
            "best_ev_today": best_ev,
        }

    def get_overall_stats(self) -> dict:
        perf   = self.data["performance"]
        total  = perf["total_wins"] + perf["total_losses"]
        roi    = (perf["total_returned"] - perf["total_staked"]) / perf["total_staked"] * 100 \
                 if perf["total_staked"] > 0 else 0.0
        return {
            "total_sent":   perf["total_sent"],
            "total_wins":   perf["total_wins"],
            "total_losses": perf["total_losses"],
            "win_rate":     perf["total_wins"] / total * 100 if total > 0 else 0,
            "roi":          roi,
            "staked":       perf["total_staked"],
            "returned":     perf["total_returned"],
        }

    def get_recent_tips(self, n: int = 10) -> List[dict]:
        return list(reversed(self.data["sent_tips"][-n:]))
