"""
Scanner — Busca partidas e odds via Claude API com web search
=============================================================
CORREÇÃO: usa AsyncAnthropic para não bloquear o event loop.
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict
from anthropic import AsyncAnthropic
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
        # AsyncAnthropic — não bloqueia o event loop
        self.client = AsyncAnthropic(api_key=api_key)

    async def _call_claude(self, prompt: str, use_search: bool = False) -> str:
        """Chama a API do Claude de forma assíncrona."""
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }
        if use_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

        response = await self.client.messages.create(**kwargs)
        return "".join(
            b.text for b in response.content if hasattr(b, "text") and b.text
        )

    def _parse_json(self, raw: str) -> dict:
        """Extrai e parseia JSON da resposta, tolerante a texto extra."""
        clean = re.sub(r"```json|```", "", raw).strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start == -1 or end <= 0:
            raise ValueError(f"JSON não encontrado. Resposta recebida: {raw[:300]}")
        return json.loads(clean[start:end])

    async def scan(
        self,
        leagues: List[str],
        max_matches: int = 10,
        banca: float = 1000.0,
        kelly_fraction: float = 0.25,
    ) -> List[Dict]:
        """
        Scan completo:
        1. Busca partidas e odds via web search
        2. Roda motor matemático (Poisson + Kelly + EV)
        3. Retorna oportunidades ordenadas por EV
        """
        log.info(f"Scan iniciado — {len(leagues)} ligas | banca R${banca:.0f} | kelly {kelly_fraction*100:.0f}%")

        # ── ETAPA 1: buscar dados ─────────────────────────────────────────────
        prompt = FETCH_PROMPT.format(
            datetime=datetime.now().strftime("%d/%m/%Y %H:%M"),
            leagues=", ".join(leagues),
            max_matches=max_matches,
        )

        try:
            raw = await self._call_claude(prompt, use_search=True)
            log.info(f"Resposta recebida ({len(raw)} chars)")
            data = self._parse_json(raw)
            matches = data.get("matches", [])
            log.info(f"{len(matches)} partidas encontradas no scan")
        except json.JSONDecodeError as e:
            log.error(f"Erro ao parsear JSON: {e}")
            return []
        except Exception as e:
            log.error(f"Erro ao buscar partidas: {e}", exc_info=True)
            return []

        # ── ETAPA 2: motor matemático ─────────────────────────────────────────
        opportunities = []
        for match in matches:
            try:
                # Validar campos mínimos
                required = ["homeTeam", "awayTeam", "homeAvgGoalsScored",
                            "homeAvgGoalsConceded", "awayAvgGoalsScored", "awayAvgGoalsConceded"]
                if not all(match.get(f) for f in required):
                    log.warning(f"Dados incompletos para {match.get('homeTeam','?')} vs {match.get('awayTeam','?')}")
                    continue

                markets = analyze_match(match, banca, kelly_fraction)
                if not markets:
                    continue

                best = markets[0]
                if best.ev_pct <= 0:
                    log.info(f"  ✗ {match['homeTeam']} vs {match['awayTeam']} — sem EV+ (melhor: {best.ev_pct:.1f}%)")
                    continue

                # Análise qualitativa (sem web search para economizar tokens)
                ai_analysis = await self._get_ai_analysis(match, markets[:3])

                opp = {
                    "home_team":    match["homeTeam"],
                    "away_team":    match["awayTeam"],
                    "league":       match["league"],
                    "date":         match.get("date", "?"),
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
                    "all_markets": [
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
                opportunities.append(opp)
                log.info(f"  ✓ {match['homeTeam']} vs {match['awayTeam']} — {best.label} | EV {best.ev_pct:.1f}% | Apostar R${best.stake:.2f}")

            except Exception as e:
                log.warning(f"Erro ao processar {match.get('homeTeam','?')} vs {match.get('awayTeam','?')}: {e}")

        opportunities.sort(key=lambda x: x["ev_pct"], reverse=True)
        log.info(f"Scan finalizado. {len(opportunities)} oportunidades com EV positivo")
        return opportunities

    async def _get_ai_analysis(self, match: dict, top_markets) -> str:
        """Análise qualitativa — texto curto para a mensagem do Telegram."""
        try:
            opps_text = "\n".join(
                f"{i+1}. {m.label}: EV {m.ev_pct:.1f}% | Edge {m.edge_pct:.1f}% | "
                f"Odd {m.odds:.2f} | Prob {m.real_prob*100:.1f}%"
                for i, m in enumerate(top_markets)
            )
            prompt = ANALYSIS_PROMPT.format(
                home=match["homeTeam"],
                away=match["awayTeam"],
                league=match.get("league", "?"),
                date=match.get("date", "?"),
                opportunities=opps_text,
                context=match.get("context", "N/A"),
            )
            return await self._call_claude(prompt, use_search=False)
        except Exception as e:
            log.warning(f"Erro na análise qualitativa: {e}")
            return "Análise indisponível."
