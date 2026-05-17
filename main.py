"""
BETTING AI ENGINE — Sistema Autônomo 24/7
Arquivo único — sem subpastas, sem imports externos
"""

import asyncio
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ── CRIAR PASTAS ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── CARREGAR .env LOCAL (opcional) ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    def __init__(self):
        self.anthropic_api_key  = os.getenv("ANTHROPIC_API_KEY", "")
        self.telegram_token     = os.getenv("TELEGRAM_TOKEN", "")
        self.telegram_chat_id   = os.getenv("TELEGRAM_CHAT_ID", "")
        self.banca              = float(os.getenv("BANCA", "1000"))
        self.stop_loss_pct      = float(os.getenv("STOP_LOSS_PCT", "5"))
        self.kelly_fraction     = float(os.getenv("KELLY_FRACTION", "0.25"))
        self.min_ev_pct         = float(os.getenv("MIN_EV_PCT", "4"))
        self.scan_interval_min  = int(os.getenv("SCAN_INTERVAL_MIN", "180"))
        self.max_matches        = int(os.getenv("MAX_MATCHES", "10"))
        self.summary_hour       = int(os.getenv("SUMMARY_HOUR", "8"))
        self.leagues            = os.getenv(
            "LEAGUES",
            "Premier League,La Liga,Brasileirão Série A,Champions League,Serie A"
        ).split(",")

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

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR MATEMÁTICO — POISSON + KELLY + EV
# ═══════════════════════════════════════════════════════════════════════════════

def poisson_prob(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = -lam + k * math.log(lam)
    for i in range(1, k + 1):
        log_p -= math.log(i)
    return math.exp(log_p)

def match_probabilities(lh: float, la: float, max_g: int = 8) -> Dict:
    home = draw = away = over25 = btts = 0.0
    for h in range(max_g + 1):
        ph = poisson_prob(lh, h)
        for a in range(max_g + 1):
            p = ph * poisson_prob(la, a)
            if h > a:    home  += p
            elif h == a: draw  += p
            else:        away  += p
            if h + a > 2.5:       over25 += p
            if h > 0 and a > 0:   btts   += p
    return {"home": home, "draw": draw, "away": away,
            "over25": over25, "under25": 1 - over25, "btts": btts}

def calc_ev(prob: float, odds: float) -> float:
    return prob * (odds - 1) - (1 - prob)

def calc_edge(prob: float, odds: float) -> float:
    return (prob - 1 / odds) * 100

def calc_kelly(prob: float, odds: float, frac: float = 0.25) -> float:
    b = odds - 1
    if b <= 0: return 0.0
    return max(0.0, ((b * prob - (1 - prob)) / b) * frac)

def classify(ev: float, edge: float):
    if ev > 0.12 and edge > 5:  return "ALTO VALOR",     "🟢"
    if ev > 0.06 and edge > 2:  return "BOM VALOR",      "🔵"
    if ev > 0.01 and edge > 0:  return "VALOR MARGINAL", "🟡"
    if ev >= 0:                  return "SEM VANTAGEM",   "🟠"
    return "EVITAR", "🔴"

MARKET_LABELS = {
    "home": "Vitória Casa", "draw": "Empate", "away": "Vitória Fora",
    "over25": "Over 2.5 Gols", "under25": "Under 2.5 Gols", "btts": "Ambas Marcam",
}

BETANO_PATH = {
    "home":    ("Principais", "Resultado Final",      "1 — Vitória Casa"),
    "draw":    ("Principais", "Resultado Final",      "X — Empate"),
    "away":    ("Principais", "Resultado Final",      "2 — Vitória Visitante"),
    "over25":  ("Gols",       "Total de Gols",        "Mais de 2.5"),
    "under25": ("Gols",       "Total de Gols",        "Menos de 2.5"),
    "btts":    ("Gols",       "Ambas Equipes Marcam", "Sim"),
}

def analyze_match(match: dict, banca: float, kelly_frac: float) -> List[Dict]:
    lh = match["homeAvgGoalsScored"] * match["awayAvgGoalsConceded"]
    la = match["awayAvgGoalsScored"] * match["homeAvgGoalsConceded"]
    probs = match_probabilities(lh, la)
    odds_map = match.get("odds", {})
    results = []
    for key, prob in probs.items():
        odds = odds_map.get(key)
        if not odds or odds <= 1.01:
            continue
        ev    = calc_ev(prob, odds)
        edge  = calc_edge(prob, odds)
        kf    = calc_kelly(prob, odds, kelly_frac)
        lbl, emoji = classify(ev, edge)
        results.append({
            "key": key, "label": MARKET_LABELS.get(key, key),
            "prob": prob, "implied": 1/odds, "odds": odds,
            "ev": ev, "ev_pct": ev * 100, "edge_pct": edge,
            "kelly": kf, "stake": banca * kf,
            "cls_label": lbl, "cls_color": emoji,
        })
    return sorted(results, key=lambda x: x["ev_pct"], reverse=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE API
# ═══════════════════════════════════════════════════════════════════════════════

async def call_claude(api_key: str, prompt: str, use_search: bool = False) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=api_key)
    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    resp = await client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if hasattr(b, "text") and b.text)

def parse_json(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    s, e = clean.find("{"), clean.rfind("}") + 1
    if s == -1 or e <= 0:
        raise ValueError(f"JSON não encontrado. Recebido: {raw[:200]}")
    return json.loads(clean[s:e])

FETCH_PROMPT = """Você é um sistema de coleta de dados para apostas esportivas.
Data/hora: {dt}

Busque na web partidas de futebol das PRÓXIMAS 24-48 HORAS nas ligas: {leagues}

Retorne SOMENTE JSON puro, zero texto fora, zero markdown:

{{"matches":[{{"homeTeam":"str","awayTeam":"str","league":"str","date":"DD/MM HH:MM","homeAvgGoalsScored":1.5,"homeAvgGoalsConceded":1.1,"awayAvgGoalsScored":1.2,"awayAvgGoalsConceded":1.3,"odds":{{"home":2.10,"draw":3.30,"away":3.50,"over25":1.85,"under25":1.95,"btts":1.75}},"context":"max 100 chars"}}]}}

REGRAS: médias últimos 5 jogos, odds reais Betano/Bet365/Pinnacle, máx {max_m} partidas, null se sem odd, SOMENTE JSON."""

ANALYSIS_PROMPT = """Analista quantitativo de apostas. Direto, técnico, sem linguagem genérica.

JOGO: {home} vs {away} — {league} — {date}
OPORTUNIDADES:
{opps}
Contexto: {ctx}

3 partes, máx 150 palavras:
1. VANTAGEM MATEMÁTICA
2. RISCO PRINCIPAL  
3. ENTRADA RECOMENDADA (mercado + odd mínima)"""

async def scan(api_key: str, leagues: List[str], max_matches: int,
               banca: float, kelly_frac: float) -> List[Dict]:
    log.info(f"Scan iniciado — {len(leagues)} ligas")
    prompt = FETCH_PROMPT.format(
        dt=datetime.now().strftime("%d/%m/%Y %H:%M"),
        leagues=", ".join(leagues),
        max_m=max_matches,
    )
    try:
        raw = await call_claude(api_key, prompt, use_search=True)
        data = parse_json(raw)
        matches = data.get("matches", [])
        log.info(f"{len(matches)} partidas encontradas")
    except Exception as e:
        log.error(f"Erro ao buscar partidas: {e}")
        return []

    opportunities = []
    for m in matches:
        try:
            required = ["homeTeam","awayTeam","homeAvgGoalsScored",
                        "homeAvgGoalsConceded","awayAvgGoalsScored","awayAvgGoalsConceded"]
            if not all(m.get(f) for f in required):
                continue

            markets = analyze_match(m, banca, kelly_frac)
            if not markets or markets[0]["ev_pct"] <= 0:
                log.info(f"  ✗ {m['homeTeam']} vs {m['awayTeam']} — sem EV+")
                continue

            best = markets[0]

            # Análise qualitativa
            try:
                opps_txt = "\n".join(
                    f"{i+1}. {mk['label']}: EV {mk['ev_pct']:.1f}% | "
                    f"Edge {mk['edge_pct']:.1f}% | Odd {mk['odds']:.2f} | Prob {mk['prob']*100:.1f}%"
                    for i, mk in enumerate(markets[:3])
                )
                ai_txt = await call_claude(api_key, ANALYSIS_PROMPT.format(
                    home=m["homeTeam"], away=m["awayTeam"],
                    league=m.get("league","?"), date=m.get("date","?"),
                    opps=opps_txt, ctx=m.get("context","N/A")
                ))
            except Exception:
                ai_txt = "Análise indisponível."

            opp = {
                "home_team": m["homeTeam"], "away_team": m["awayTeam"],
                "league": m.get("league","?"), "date": m.get("date","?"),
                "context": m.get("context",""),
                "market_key": best["key"], "market_label": best["label"],
                "odds": best["odds"], "real_prob": best["prob"],
                "implied_prob": best["implied"], "ev_pct": best["ev_pct"],
                "edge_pct": best["edge_pct"], "kelly": best["kelly"],
                "stake": best["stake"], "cls_label": best["cls_label"],
                "cls_color": best["cls_color"],
                "all_markets": markets,
                "ai_analysis": ai_txt,
                "scanned_at": datetime.now().isoformat(),
            }
            opportunities.append(opp)
            log.info(f"  ✓ {m['homeTeam']} vs {m['awayTeam']} — {best['label']} EV:{best['ev_pct']:.1f}% Apostar:R${best['stake']:.2f}")

        except Exception as e:
            log.warning(f"Erro em {m.get('homeTeam','?')}: {e}")

    opportunities.sort(key=lambda x: x["ev_pct"], reverse=True)
    log.info(f"Scan finalizado. {len(opportunities)} oportunidades EV+")
    return opportunities

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════

async def tg_send(token: str, chat_id: str, text: str) -> bool:
    import aiohttp
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:4096],
                "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.post(url, json=payload) as r:
                body = await r.json()
                if r.status == 200 and body.get("ok"):
                    return True
                log.error(f"Telegram {r.status}: {body.get('description','?')}")
                if r.status == 401:
                    log.error("TELEGRAM_TOKEN inválido!")
                elif "chat not found" in str(body):
                    log.error("TELEGRAM_CHAT_ID inválido!")
                return False
    except Exception as e:
        log.error(f"Erro Telegram: {e}")
        return False

async def send_opportunity(token: str, chat_id: str, opp: Dict) -> bool:
    nav = BETANO_PATH.get(opp["market_key"],
          ("Principais", opp["market_label"], opp["market_label"]))
    tab, section, option = nav
    retorno = opp["stake"] * opp["odds"]
    lucro   = opp["stake"] * (opp["odds"] - 1)
    odd_min = round(opp["odds"] * 0.97, 2)

    alt = [m for m in opp.get("all_markets",[])
           if m["ev_pct"] > 0 and m["label"] != opp["market_label"]][:2]
    alt_txt = "\n".join(
        f"   • {m['label']} @ {m['odds']:.2f} — EV {m['ev_pct']:.1f}%"
        for m in alt
    ) or "   — Apenas este mercado tem EV positivo"

    msg = (
        f"🎯 <b>OPORTUNIDADE DETECTADA</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⚽ <b>{opp['home_team']} vs {opp['away_team']}</b>\n"
        f"📅 {opp['date']} | {opp['league']}\n\n"
        f"📊 <b>MERCADO:</b> {opp['market_label']}\n"
        f"💰 <b>ODD:</b> {opp['odds']:.2f}  |  Mínima: {odd_min}\n"
        f"📈 <b>EV:</b> +{opp['ev_pct']:.1f}%  |  Edge: {opp['edge_pct']:.1f}%\n"
        f"🏦 <b>APOSTAR:</b> R$ {opp['stake']:.2f}\n"
        f"💵 Retorno: R$ {retorno:.2f}  |  Lucro: R$ {lucro:.2f}\n"
        f"{opp['cls_color']} {opp['cls_label']}\n\n"
        f"📍 <b>BETANO:</b>\n"
        f"<code>Futebol → {opp['league']}\n"
        f"→ {opp['home_team']} vs {opp['away_team']}\n"
        f"→ {tab} → {section} → {option}</code>\n\n"
        f"⚠️ Odd mínima: <b>{odd_min}</b>  ⏱ Execute em até 5 min\n\n"
        f"<b>Outros mercados EV+:</b>\n{alt_txt}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧮 Prob real: {opp['real_prob']*100:.1f}%  "
        f"Impl: {opp['implied_prob']*100:.1f}%\n"
        f"🕐 {datetime.now().strftime('%H:%M:%S')}"
    )
    return await tg_send(token, chat_id, msg)

async def send_analysis(token: str, chat_id: str, opp: Dict) -> bool:
    if not opp.get("ai_analysis"):
        return False
    msg = (
        f"🤖 <b>ANÁLISE DA IA</b>\n"
        f"{opp['home_team']} vs {opp['away_team']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{opp['ai_analysis'][:3000]}"
    )
    return await tg_send(token, chat_id, msg)

async def send_startup(token: str, chat_id: str, cfg: Config) -> bool:
    stop_val = cfg.banca * cfg.stop_loss_pct / 100
    msg = (
        f"🟢 <b>BETTING AI ENGINE — ONLINE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 Banca: R$ {cfg.banca:.2f}\n"
        f"🛑 Stop loss: R$ {stop_val:.2f} ({cfg.stop_loss_pct}%)\n"
        f"📈 EV mínimo: {cfg.min_ev_pct}%\n"
        f"🔄 Scan a cada {cfg.scan_interval_min} min\n"
        f"⚽ Ligas: {', '.join(cfg.leagues)}\n\n"
        f"Sistema ativo. Enviando alertas automaticamente.\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    return await tg_send(token, chat_id, msg)

async def send_summary(token: str, chat_id: str, stats: Dict) -> bool:
    msg = (
        f"📊 <b>RESUMO DO DIA</b> — {datetime.now().strftime('%d/%m/%Y')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📤 Sinais: {stats.get('sent',0)}\n"
        f"📈 Melhor EV: {stats.get('best_ev',0):.1f}%\n\n"
        f"Sistema continua monitorando 24h."
    )
    return await tg_send(token, chat_id, msg)

async def send_error(token: str, chat_id: str, error: str) -> bool:
    msg = (
        f"⚠️ <b>ERRO NO SISTEMA</b>\n"
        f"<code>{error[:400]}</code>\n"
        f"🕐 {datetime.now().strftime('%d/%m %H:%M')}\n"
        f"Recuperando no próximo ciclo."
    )
    return await tg_send(token, chat_id, msg)

# ═══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA / HISTÓRICO
# ═══════════════════════════════════════════════════════════════════════════════

class Memory:
    def __init__(self, path="data/history.json"):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"tips": [], "perf": {"sent":0,"wins":0,"losses":0,"staked":0.0,"returned":0.0}}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Erro ao salvar histórico: {e}")

    def already_sent_today(self, match_key: str, market_key: str) -> bool:
        today = date.today().isoformat()
        return any(
            t["match"] == match_key and t["market"] == market_key and t["date"] == today
            for t in self.data["tips"]
        )

    def record(self, match_key: str, opp: Dict):
        self.data["tips"].append({
            "id": len(self.data["tips"]) + 1,
            "match": match_key, "market": opp["market_key"],
            "league": opp["league"], "odds": opp["odds"],
            "stake": opp["stake"], "ev_pct": opp["ev_pct"],
            "date": date.today().isoformat(),
            "sent_at": datetime.now().isoformat(),
            "outcome": None,
        })
        self.data["perf"]["sent"] += 1
        if len(self.data["tips"]) > 500:
            self.data["tips"] = self.data["tips"][-500:]
        self._save()

    def get_adaptive_min_ev(self, base: float) -> float:
        p = self.data["perf"]
        total = p["wins"] + p["losses"]
        if total < 20:
            return base
        wr = p["wins"] / total
        if wr > 0.58:
            return max(base * 0.85, 2.0)
        if wr < 0.38:
            return base * 1.25
        return base

    def daily_stats(self) -> Dict:
        today = date.today().isoformat()
        today_tips = [t for t in self.data["tips"] if t["date"] == today]
        return {
            "sent": len(today_tips),
            "best_ev": max((t["ev_pct"] for t in today_tips), default=0.0),
        }

# ═══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

async def run_cycle(cfg: Config, memory: Memory):
    log.info("=" * 50)
    log.info(f"Ciclo — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    try:
        opps = await scan(
            cfg.anthropic_api_key, cfg.leagues,
            cfg.max_matches, cfg.banca, cfg.kelly_fraction
        )

        if not opps:
            log.info("Sem oportunidades EV+")
            await tg_send(
                cfg.telegram_token, cfg.telegram_chat_id,
                f"🔍 Scan {datetime.now().strftime('%d/%m %H:%M')} — Sem oportunidades acima do limiar."
            )
            return

        min_ev = memory.get_adaptive_min_ev(cfg.min_ev_pct)
        filtered = [o for o in opps if o["ev_pct"] >= min_ev]
        log.info(f"{len(filtered)} acima do EV mínimo ({min_ev:.1f}%)")

        sent = 0
        for opp in filtered:
            key = f"{opp['home_team']}-{opp['away_team']}"
            if memory.already_sent_today(key, opp["market_key"]):
                log.info(f"Já enviado hoje: {key}")
                continue
            ok = await send_opportunity(cfg.telegram_token, cfg.telegram_chat_id, opp)
            if ok:
                await asyncio.sleep(1)
                await send_analysis(cfg.telegram_token, cfg.telegram_chat_id, opp)
                memory.record(key, opp)
                sent += 1
                await asyncio.sleep(2)

        log.info(f"{sent} alertas enviados")

        now = datetime.now()
        if now.hour == cfg.summary_hour and now.minute < 10:
            await send_summary(cfg.telegram_token, cfg.telegram_chat_id, memory.daily_stats())

    except Exception as e:
        log.error(f"Erro no ciclo: {e}", exc_info=True)
        try:
            await send_error(cfg.telegram_token, cfg.telegram_chat_id, str(e))
        except Exception:
            pass


async def main():
    log.info("=" * 60)
    log.info("🤖 BETTING AI ENGINE — INICIANDO")
    log.info("=" * 60)

    cfg = Config()
    try:
        cfg.validate()
    except ValueError as e:
        log.error(str(e))
        log.error("Verifique as variáveis de ambiente no Railway!")
        sys.exit(1)

    log.info(f"Banca: R${cfg.banca} | EV mín: {cfg.min_ev_pct}% | Scan: {cfg.scan_interval_min}min")

    memory = Memory()

    log.info("Enviando mensagem de startup...")
    ok = await send_startup(cfg.telegram_token, cfg.telegram_chat_id, cfg)
    if ok:
        log.info("✅ Telegram funcionando!")
    else:
        log.error("❌ Falha no Telegram — verifique TOKEN e CHAT_ID")

    cycle = 0
    while True:
        cycle += 1
        log.info(f"─── Ciclo #{cycle} ───")
        await run_cycle(cfg, memory)
        log.info(f"Aguardando {cfg.scan_interval_min} minutos...")
        await asyncio.sleep(cfg.scan_interval_min * 60)


if __name__ == "__main__":
    asyncio.run(main())
