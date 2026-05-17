"""
BETTING AI ENGINE — Sistema Autônomo 24/7
Usa apenas bibliotecas padrão do Python para HTTP (sem aiohttp).
"""

import asyncio
import json
import logging
import math
import os
import re
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, date
from typing import List, Dict

# ── CRIAR PASTAS ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── CARREGAR .env LOCAL ───────────────────────────────────────────────────────
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
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.telegram_token    = os.getenv("TELEGRAM_TOKEN", "")
        self.telegram_chat_id  = os.getenv("TELEGRAM_CHAT_ID", "")
        self.banca             = float(os.getenv("BANCA", "1000"))
        self.stop_loss_pct     = float(os.getenv("STOP_LOSS_PCT", "5"))
        self.kelly_fraction    = float(os.getenv("KELLY_FRACTION", "0.25"))
        self.min_ev_pct        = float(os.getenv("MIN_EV_PCT", "4"))
        self.scan_interval_min = int(os.getenv("SCAN_INTERVAL_MIN", "180"))
        self.max_matches       = int(os.getenv("MAX_MATCHES", "10"))
        self.summary_hour      = int(os.getenv("SUMMARY_HOUR", "8"))
        self.leagues           = os.getenv(
            "LEAGUES",
            "Premier League,La Liga,Brasileirão Série A,Champions League,Serie A"
        ).split(",")

    def validate(self):
        errors = []
        if not self.anthropic_api_key: errors.append("ANTHROPIC_API_KEY não definida")
        if not self.telegram_token:    errors.append("TELEGRAM_TOKEN não definida")
        if not self.telegram_chat_id:  errors.append("TELEGRAM_CHAT_ID não definida")
        if errors:
            raise ValueError("Configuração inválida:\n" + "\n".join(f"  • {e}" for e in errors))

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR MATEMÁTICO
# ═══════════════════════════════════════════════════════════════════════════════
def poisson_prob(lam: float, k: int) -> float:
    if lam <= 0: return 1.0 if k == 0 else 0.0
    log_p = -lam + k * math.log(lam)
    for i in range(1, k + 1): log_p -= math.log(i)
    return math.exp(log_p)

def match_probs(lh: float, la: float, max_g: int = 8) -> Dict:
    home = draw = away = over25 = btts = 0.0
    for h in range(max_g + 1):
        ph = poisson_prob(lh, h)
        for a in range(max_g + 1):
            p = ph * poisson_prob(la, a)
            if h > a:          home  += p
            elif h == a:       draw  += p
            else:              away  += p
            if h + a > 2.5:    over25 += p
            if h > 0 and a > 0: btts  += p
    return {"home": home, "draw": draw, "away": away,
            "over25": over25, "under25": 1 - over25, "btts": btts}

def calc_ev(prob, odds):    return prob * (odds - 1) - (1 - prob)
def calc_edge(prob, odds):  return (prob - 1 / odds) * 100
def calc_kelly(prob, odds, frac=0.25):
    b = odds - 1
    return max(0.0, ((b * prob - (1 - prob)) / b) * frac) if b > 0 else 0.0

def classify(ev, edge):
    if ev > 0.12 and edge > 5:  return "ALTO VALOR",     "🟢"
    if ev > 0.06 and edge > 2:  return "BOM VALOR",      "🔵"
    if ev > 0.01 and edge > 0:  return "VALOR MARGINAL", "🟡"
    if ev >= 0:                  return "SEM VANTAGEM",   "🟠"
    return "EVITAR", "🔴"

MARKET_LABELS = {
    "home":"Vitória Casa","draw":"Empate","away":"Vitória Fora",
    "over25":"Over 2.5","under25":"Under 2.5","btts":"Ambas Marcam",
}
BETANO_PATH = {
    "home":    ("Principais","Resultado Final","1 — Vitória Casa"),
    "draw":    ("Principais","Resultado Final","X — Empate"),
    "away":    ("Principais","Resultado Final","2 — Vitória Visitante"),
    "over25":  ("Gols","Total de Gols","Mais de 2.5"),
    "under25": ("Gols","Total de Gols","Menos de 2.5"),
    "btts":    ("Gols","Ambas Equipes Marcam","Sim"),
}

def analyze_match(match: dict, banca: float, kfrac: float) -> List[Dict]:
    lh = match["homeAvgGoalsScored"] * match["awayAvgGoalsConceded"]
    la = match["awayAvgGoalsScored"] * match["homeAvgGoalsConceded"]
    probs = match_probs(lh, la)
    results = []
    for key, prob in probs.items():
        odds = (match.get("odds") or {}).get(key)
        if not odds or odds <= 1.01: continue
        ev   = calc_ev(prob, odds)
        edge = calc_edge(prob, odds)
        kf   = calc_kelly(prob, odds, kfrac)
        lbl, emoji = classify(ev, edge)
        results.append({
            "key":key,"label":MARKET_LABELS.get(key,key),
            "prob":prob,"implied":1/odds,"odds":odds,
            "ev":ev,"ev_pct":ev*100,"edge_pct":edge,
            "kelly":kf,"stake":banca*kf,
            "cls_label":lbl,"cls_color":emoji,
        })
    return sorted(results, key=lambda x: x["ev_pct"], reverse=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM — usa urllib (stdlib, sem instalação)
# ═══════════════════════════════════════════════════════════════════════════════
def tg_send_sync(token: str, chat_id: str, text: str) -> bool:
    """Envia mensagem pelo Telegram usando urllib (stdlib)."""
    if not token or not chat_id:
        log.error("Telegram não configurado")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text[:4096],
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            if body.get("ok"):
                return True
            log.error(f"Telegram erro: {body.get('description','?')}")
            return False
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error(f"Telegram HTTP {e.code}: {body}")
        if e.code == 401: log.error("TELEGRAM_TOKEN inválido!")
        return False
    except Exception as e:
        log.error(f"Erro Telegram: {e}")
        return False

async def tg_send(token: str, chat_id: str, text: str) -> bool:
    """Wrapper async para não bloquear o event loop."""
    return await asyncio.get_event_loop().run_in_executor(
        None, tg_send_sync, token, chat_id, text
    )

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
    if s == -1 or e <= 0: raise ValueError(f"JSON não encontrado: {raw[:200]}")
    return json.loads(clean[s:e])

# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
FETCH_PROMPT = """Você é um sistema de coleta de dados para apostas esportivas.
Data/hora: {dt}

Busque na web partidas de futebol das PRÓXIMAS 24-48 HORAS nas ligas: {leagues}

Retorne SOMENTE JSON puro — zero texto fora, zero markdown:
{{"matches":[{{"homeTeam":"str","awayTeam":"str","league":"str","date":"DD/MM HH:MM","homeAvgGoalsScored":1.5,"homeAvgGoalsConceded":1.1,"awayAvgGoalsScored":1.2,"awayAvgGoalsConceded":1.3,"odds":{{"home":2.10,"draw":3.30,"away":3.50,"over25":1.85,"under25":1.95,"btts":1.75}},"context":"max 100 chars"}}]}}

REGRAS: médias últimos 5 jogos, odds reais Betano/Bet365/Pinnacle, máx {max_m} partidas, null se sem odd, SOMENTE JSON."""

ANALYSIS_PROMPT = """Analista quantitativo de apostas. Direto, técnico.

{home} vs {away} — {league} — {date}
{opps}
Contexto: {ctx}

3 partes, máx 150 palavras:
1. VANTAGEM MATEMÁTICA
2. RISCO PRINCIPAL
3. ENTRADA RECOMENDADA (mercado + odd mínima)"""

async def scan(api_key, leagues, max_matches, banca, kfrac) -> List[Dict]:
    log.info(f"Scan — {len(leagues)} ligas")
    prompt = FETCH_PROMPT.format(
        dt=datetime.now().strftime("%d/%m/%Y %H:%M"),
        leagues=", ".join(leagues), max_m=max_matches
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
            if not all(m.get(f) for f in required): continue
            markets = analyze_match(m, banca, kfrac)
            if not markets or markets[0]["ev_pct"] <= 0:
                log.info(f"  ✗ {m['homeTeam']} vs {m['awayTeam']} — sem EV+")
                continue
            best = markets[0]
            # Análise qualitativa
            try:
                opps_txt = "\n".join(
                    f"{i+1}. {mk['label']}: EV {mk['ev_pct']:.1f}% | "
                    f"Edge {mk['edge_pct']:.1f}% | Odd {mk['odds']:.2f}"
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
                "home_team":m["homeTeam"],"away_team":m["awayTeam"],
                "league":m.get("league","?"),"date":m.get("date","?"),
                "context":m.get("context",""),
                "market_key":best["key"],"market_label":best["label"],
                "odds":best["odds"],"real_prob":best["prob"],
                "implied_prob":best["implied"],"ev_pct":best["ev_pct"],
                "edge_pct":best["edge_pct"],"kelly":best["kelly"],
                "stake":best["stake"],"cls_label":best["cls_label"],
                "cls_color":best["cls_color"],"all_markets":markets,
                "ai_analysis":ai_txt,
            }
            opportunities.append(opp)
            log.info(f"  ✓ {m['homeTeam']} vs {m['awayTeam']} — {best['label']} EV:{best['ev_pct']:.1f}% R${best['stake']:.2f}")
        except Exception as e:
            log.warning(f"Erro em {m.get('homeTeam','?')}: {e}")

    opportunities.sort(key=lambda x: x["ev_pct"], reverse=True)
    log.info(f"Scan finalizado. {len(opportunities)} oportunidades EV+")
    return opportunities

# ═══════════════════════════════════════════════════════════════════════════════
# MENSAGENS TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
async def send_opportunity(token, chat_id, opp) -> bool:
    nav = BETANO_PATH.get(opp["market_key"], ("Principais", opp["market_label"], opp["market_label"]))
    tab, section, option = nav
    retorno = opp["stake"] * opp["odds"]
    lucro   = opp["stake"] * (opp["odds"] - 1)
    odd_min = round(opp["odds"] * 0.97, 2)
    alt = [m for m in opp.get("all_markets",[]) if m["ev_pct"] > 0 and m["label"] != opp["market_label"]][:2]
    alt_txt = "\n".join(f"   • {m['label']} @ {m['odds']:.2f} — EV {m['ev_pct']:.1f}%" for m in alt) \
              or "   — Apenas este mercado tem EV positivo"
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
        f"→ {tab} → {section}\n"
        f"→ {option}</code>\n\n"
        f"⚠️ Odd mínima: <b>{odd_min}</b>  ⏱ Até 5 min\n\n"
        f"<b>Outros mercados EV+:</b>\n{alt_txt}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧮 Prob real: {opp['real_prob']*100:.1f}%  Impl: {opp['implied_prob']*100:.1f}%\n"
        f"🕐 {datetime.now().strftime('%H:%M:%S')}"
    )
    return await tg_send(token, chat_id, msg)

async def send_analysis(token, chat_id, opp) -> bool:
    if not opp.get("ai_analysis"): return False
    msg = (f"🤖 <b>ANÁLISE DA IA</b>\n"
           f"{opp['home_team']} vs {opp['away_team']}\n"
           f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
           f"{opp['ai_analysis'][:3000]}")
    return await tg_send(token, chat_id, msg)

async def send_startup(token, chat_id, cfg) -> bool:
    msg = (
        f"🟢 <b>BETTING AI ENGINE — ONLINE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 Banca: R$ {cfg.banca:.2f}\n"
        f"🛑 Stop loss: R$ {cfg.banca * cfg.stop_loss_pct / 100:.2f} ({cfg.stop_loss_pct}%)\n"
        f"📈 EV mínimo: {cfg.min_ev_pct}%\n"
        f"🔄 Scan a cada {cfg.scan_interval_min} min\n"
        f"⚽ Ligas: {', '.join(cfg.leagues)}\n\n"
        f"Sistema ativo 24h. Alertas automáticos.\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    return await tg_send(token, chat_id, msg)

async def send_no_opps(token, chat_id):
    return await tg_send(token, chat_id,
        f"🔍 Scan {datetime.now().strftime('%d/%m %H:%M')} — Sem oportunidades acima do limiar.")

async def send_error(token, chat_id, error):
    return await tg_send(token, chat_id,
        f"⚠️ <b>ERRO</b>\n<code>{error[:400]}</code>\n"
        f"🕐 {datetime.now().strftime('%d/%m %H:%M')}\nRecuperando no próximo ciclo.")

# ═══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA
# ═══════════════════════════════════════════════════════════════════════════════
class Memory:
    def __init__(self, path="data/history.json"):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f: return json.load(f)
            except Exception: pass
        return {"tips":[], "perf":{"sent":0,"wins":0,"losses":0}}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e: log.error(f"Erro ao salvar: {e}")

    def already_sent_today(self, match_key, market_key) -> bool:
        today = date.today().isoformat()
        return any(t["match"]==match_key and t["market"]==market_key and t["date"]==today
                   for t in self.data["tips"])

    def record(self, match_key, opp):
        self.data["tips"].append({
            "match":match_key,"market":opp["market_key"],"league":opp["league"],
            "odds":opp["odds"],"stake":opp["stake"],"ev_pct":opp["ev_pct"],
            "date":date.today().isoformat(),"outcome":None,
        })
        self.data["perf"]["sent"] += 1
        if len(self.data["tips"]) > 500: self.data["tips"] = self.data["tips"][-500:]
        self._save()

    def get_adaptive_min_ev(self, base):
        p = self.data["perf"]
        total = p["wins"] + p["losses"]
        if total < 20: return base
        wr = p["wins"] / total
        if wr > 0.58: return max(base * 0.85, 2.0)
        if wr < 0.38: return base * 1.25
        return base

# ═══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
async def run_cycle(cfg: Config, memory: Memory):
    log.info("=" * 50)
    log.info(f"Ciclo — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    try:
        opps = await scan(cfg.anthropic_api_key, cfg.leagues,
                          cfg.max_matches, cfg.banca, cfg.kelly_fraction)
        if not opps:
            log.info("Sem oportunidades EV+")
            await send_no_opps(cfg.telegram_token, cfg.telegram_chat_id)
            return

        min_ev   = memory.get_adaptive_min_ev(cfg.min_ev_pct)
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

    except Exception as e:
        log.error(f"Erro no ciclo: {e}", exc_info=True)
        try: await send_error(cfg.telegram_token, cfg.telegram_chat_id, str(e))
        except Exception: pass


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

    log.info("Testando conexão com Telegram...")
    ok = await send_startup(cfg.telegram_token, cfg.telegram_chat_id, cfg)
    if ok:
        log.info("✅ Telegram OK — mensagem de startup enviada!")
    else:
        log.error("❌ Telegram falhou — verifique TOKEN e CHAT_ID nas variáveis do Railway")

    cycle = 0
    while True:
        cycle += 1
        log.info(f"─── Ciclo #{cycle} ───")
        await run_cycle(cfg, memory)
        log.info(f"Aguardando {cfg.scan_interval_min} minutos...")
        await asyncio.sleep(cfg.scan_interval_min * 60)


if __name__ == "__main__":
    asyncio.run(main())
