"""
Scanner — Busca partidas e odds via Claude API com web search
=============================================================
Usa o modelo Claude com busca web para encontrar dados reais
de partidas e odds, depois roda o motor matemático.
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional
from anthropic import Anthropic
from engine.math_engine import analyze_match

log = logging.getLogger(__name__)


FETCH_PROMPT = """Você é um sistema de coleta de dados para apostas esportivas.
Data/hora atual: {datetime}

Busque na web partidas de futebol das PRÓXIMAS 24-48 HORAS nas ligas: {leagues}

Retorne SOMENTE JSON puro — zero texto fora do JSON, zero markdown.

{{
  "matches": [
    {{
      "homeTeam": "string",
      "awayTeam": "string",
      "league": "string",
      "date": "DD/MM HH:MM",
      "homeAvgGoalsScored": 1.5,
      "homeAvgGoalsConceded": 1.1,
      "awayAvgGoalsScored": 1.2,
      "awayAvgGoalsConceded": 1.3,
      "odds": {{
        "home": 2.10,
        "draw": 3.30,
        "away": 3.50,
        "over25": 1.85,
        "under25": 1.95,
        "btts": 1.75
      }},
      "context": "Contexto: forma recente, lesões importantes (max 120 chars)"
    }}
  ]
}}

REGRAS OBRIGATÓRIAS:
- Médias dos ÚLTIMOS 5 jogos de cada time
- Odds reais da Betano, Bet365, Betfair ou Pinnacle
- Máximo {max_matches} partidas
- Se não tiver odd de algum mercado, use null
- SOMENTE JSON puro, sem nenhum texto fora"""


ANALYSIS_PROMPT = """Analista quantitativo de apostas esportivas. Técnico, direto, sem linguagem genérica.

JOGO: {home} vs {away} — {league} — {date}

OPORTUNIDADES DETECTADAS (por EV):
{opportunities}

Contexto: {context}

Responda em 3 partes curtas (máx 150 palavras total):
1. VANTAGEM MATEMÁTICA: sustentação do EV
2. RISCO PRINCIPAL: o que pode invalidar
3. CONCLUSÃO: apostar ou não, mercado e odd mínima"""


class Scanner:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def _call_claude(self, prompt: str, use_search: bool = False) -> str:
        """Chama a API do Claude, com ou sem web search."""
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }
        if use_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

        response = self.client.messages.create(**kwargs)
        return "".join(b.text for b in response.content if hasattr(b, "text"))

    def _parse_json(self, raw: str) -> dict:
        """Extrai e parseia JSON da resposta do Claude."""
        clean = re.sub(r"```json|```", "", raw).strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON não encontrado na resposta")
        return json.loads(clean[start:end])

    async def scan(self, leagues: List[str], max_matches: int = 10,
                   banca: float = 1000, kelly_fraction: float = 0.25) -> List[Dict]:
        """
        Scan completo: busca partidas → analisa → retorna oportunidades ordenadas por EV.
        """
        log.info(f"Iniciando scan — {len(leagues)} ligas, max {max_matches} partidas")

        # 1. Buscar dados das partidas
        prompt = FETCH_PROMPT.format(
            datetime=datetime.now().strftime("%d/%m/%Y %H:%M"),
            leagues=", ".join(leagues),
            max_matches=max_matches,
        )

        try:
            raw = self._call_claude(prompt, use_search=True)
            data = self._parse_json(raw)
            matches = data.get("matches", [])
            log.info(f"{len(matches)} partidas encontradas")
        except Exception as e:
            log.error(f"Erro ao buscar partidas: {e}")
            return []

        # 2. Rodar motor matemático em cada partida
        opportunities = []
        for match in matches:
            try:
                markets = analyze_match(match, banca, kelly_fraction)
                best    = markets[0] if markets else None

                if not best or best.ev_pct <= 0:
                    continue

                # Gerar análise qualitativa da IA
                ai_analysis = self._get_ai_analysis(match, markets[:3])

                opportunity = {
                    "home_team":    match["homeTeam"],
                    "away_team":    match["awayTeam"],
                    "league":       match["league"],
                    "date":         match["date"],
                    "context":      match.get("context", ""),
                    "market_key":   best.key,
                    "market_label": best.label,
                    "odds":         best.odds,
                    "real_prob":    best.real_prob,
                    "implied_prob": best.implied_prob,
                    "ev":           best.ev,
                    "ev_pct":       best.ev_pct,
                    "edge_pct":     best.edge_pct,
                    "kelly_frac":   best.kelly_frac,
                    "stake":        best.stake,
                    "cls_label":    best.cls_label,
                    "cls_color":    best.cls_color,
                    "all_markets":  [
                        {
                            "label":     m.label,
                            "odds":      m.odds,
                            "ev_pct":    m.ev_pct,
                            "edge_pct":  m.edge_pct,
                            "real_prob": m.real_prob,
                        }
                        for m in markets
                    ],
                    "ai_analysis":  ai_analysis,
                    "scanned_at":   datetime.now().isoformat(),
                }
                opportunities.append(opportunity)
                log.info(f"  ✓ {match['homeTeam']} vs {match['awayTeam']} — {best.label} EV: {best.ev_pct:.1f}%")

            except Exception as e:
                log.warning(f"Erro ao processar {match.get('homeTeam','?')} vs {match.get('awayTeam','?')}: {e}")

        # Ordenar por EV
        opportunities.sort(key=lambda x: x["ev_pct"], reverse=True)
        log.info(f"Scan finalizado. {len(opportunities)} oportunidades com EV positivo")
        return opportunities

    def _get_ai_analysis(self, match: dict, top_markets) -> str:
        """Gera análise qualitativa da IA para o jogo."""
        try:
            opps_text = "\n".join(
                f"{i+1}. {m.label}: EV {m.ev_pct:.1f}% | Edge {m.edge_pct:.1f}% | "
                f"Odd {m.odds:.2f} | Prob {m.real_prob*100:.1f}%"
                for i, m in enumerate(top_markets)
            )
            prompt = ANALYSIS_PROMPT.format(
                home=match["homeTeam"],
                away=match["awayTeam"],
                league=match["league"],
                date=match["date"],
                opportunities=opps_text,
                context=match.get("context", "N/A"),
            )
            return self._call_claude(prompt, use_search=False)
        except Exception as e:
            log.warning(f"Erro na análise qualitativa: {e}")
            return "Análise indisponível."
