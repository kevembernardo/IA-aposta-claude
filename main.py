"""
BETTING AI ENGINE — Sistema Autônomo 24/7
Usa Google Gemini (100% gratuito) com Google Search integrado.
"""

import asyncio
import json
import logging
import math
import os
import re
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from typing import List, Dict, Tuple

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# executor para rodar chamadas síncronas do Gemini sem bloquear o loop
_executor = ThreadPoolExecutor(max_workers=2)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    def __init__(self):
        self.gemini_api_key    = os.getenv("GEMINI_API_KEY", "")
        self.telegram_token    = os.getenv("TELEGRAM_TOKEN", "")
        self.telegram_chat_id  = os.getenv("TELEGRAM_CHAT_ID", "")
        self.banca             = float(os.getenv("BANCA", "1000"))
        self.stop_loss_pct     = float(os.getenv("STOP_LOSS_PCT", "5"))
        self.kelly_fraction    = float(os.getenv("KELLY_FRACTION", "0.25"))
        self.min_ev_pct        = float(os.getenv("MIN_EV_PCT", "1"))
        self.scan_interval_min = int(os.getenv("SCAN_INTERVAL_MIN", "180"))
        self.max_matches       = int(os.getenv("MAX_MATCHES", "10"))
        self.debug_mode        = os.getenv("DEBUG_MODE", "true").lower() == "true"
        self.leagues           = os.getenv(
            "LEAGUES",
            "Premier League,La Liga,Brasileirão Série A,Champions League,Serie A"
        ).split(",")

    def validate(self):
        errors = []
        if not self.gemini_api_key:   errors.append("GEMINI_API_KEY não definida")
        if not self.telegram_token:   errors.append("TELEGRAM_TOKEN não definida")
        if not self.telegram_chat_id: errors.append("TELEGRAM_CHAT_ID não definida")
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
            if h > a:           home   += p
            elif h == a:        draw   += p
            else:               away   += p
            if h + a > 2.5:     over25 += p
            if h > 0 and a > 0: btts   += p
    return {"home": home, "draw": draw, "away": away,
            "over25": over25, "under25": 1 - over25, "btts": btts}

def calc_ev(prob, odds):   return prob * (odds - 1) - (1 - prob)
def calc_edge(prob, odds): return (prob - 1 / odds) * 100
def calc_kelly(prob, odds, frac=0.25):
    b = odds - 1
    return max(0.0, ((b * prob - (1 - prob)) / b) * frac) if b > 0 else 0.0

def classify(ev, edge):
    if ev > 0.12 and edge > 5: return "ALTO VALOR",     "🟢"
    if ev > 0.06 and edge > 2: return "BOM VALOR",      "🔵"
    if ev > 0.01 and edge > 0: return "VALOR MARGINAL", "🟡"
    if ev >= 0:                 return "SEM VANTAGEM",   "🟠"
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
# GEMINI API — 100% GRATUITO
# ═══════════════════════════════════════════════════════════════════════════════
def _gemini_call_sync(api_key: str, prompt: str, use_search: bool = False) -> str:
    """Chamada síncrona ao Gemini — roda em thread separada."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    if use_search:
        # Gemini com Google Search grounding — busca real na web
        model = genai.GenerativeModel("gemini-1.5-flash")
        tools = [{"google_search_retrieval": {}}]
        resp  = model.generate_content(prompt, tools=tools)
    else:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp  = model.generate_content(prompt)

    return resp.text or ""

async def call_gemini(api_key: str, prompt: str, use_search: bool = False) -> str:
    """Wrapper async — não bloqueia o event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _gemini_call_sync, api_key, prompt, use_search
    )

def parse_json(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    s, e  = clean.find("{"), clean.rfind("}") + 1
    if s == -1 or e <= 0:
        raise ValueError(f"JSON não encontrado. Resposta: {raw[:300]}")
    return json.loads(clean[s:e])

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM — urllib stdlib
# ═══════════════════════════════════════════════════════════════════════════════
def tg_send_sync(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        log.error("Telegram não configurado")
        return False
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id, "text": text[:4096],
        "parse_mode": "HTML", "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            return body.get("ok", False)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error(f"Telegram HTTP {e.code}: {body}")
        return False
    except Exception as e:
        log.error(f"Erro Telegram: {e}")
        return False

async def tg_send(token, chat_id, text) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, tg_send_sync, token, chat_id, text)

# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
FETCH_PROMPT = """Você é um coletor de dados de apostas esportivas.
Data/hora atual: {dt}

Use Google Search para encontrar partidas de futebol REAIS que acontecem HOJE ({dt_short}) ou AMANHÃ nas ligas: {leagues}

Busque também as odds atuais dessas partidas na Betano, Bet365 ou Sportingbet.
Para médias de gols, use os últimos 5 jogos de cada time (Sofascore ou FlashScore).

Responda SOMENTE com JSON puro, sem texto, sem markdown:

{{"matches":[
{{"homeTeam":"Nome do time","awayTeam":"Nome do time","league":"Liga","date":"{dt_short} HH:MM",
"homeAvgGoalsScored":1.4,"homeAvgGoalsConceded":1.0,
"awayAvgGoalsScored":1.1,"awayAvgGoalsConceded":1.3,
"odds":{{"home":2.20,"draw":3.10,"away":3.40,"over25":1.90,"under25":1.85,"btts":1.80}},
"context":"informação relevante sobre o jogo"}}
]}}

Inclua entre 5 e {max_m} partidas reais. Use null para odds não encontradas. SOMENTE JSON."""

ANALYSIS_PROMPT = """Analista de apostas esportivas. Objetivo e técnico.

{home} vs {away} — {league} — {date}
Mercados com valor matemático:
{opps}
Contexto: {ctx}

Responda em 3 partes (máx 120 palavras):
1. VANTAGEM: por que o EV é positivo aqui
2. RISCO: principal fator que pode invalidar
3. ENTRADA: mercado exato e odd mínima aceitável"""

async def scan(api_key, leagues, max_matches, banca, kfrac) -> Tuple[List[Dict], Dict]:
    log.info(f"Scan iniciado — {len(leagues)} ligas")
    now    = datetime.now()
    prompt = FETCH_PROMPT.format(
        dt=now.strftime("%d/%m/%Y %H:%M"),
        dt_short=now.strftime("%d/%m"),
        leagues=", ".join(leagues),
        max_m=max_matches,
    )
    diag = {"matches_found":0,"matches_list":[],"ev_results":[],"errors":[],"ev_plus_count":0}

    # ── Buscar dados via Gemini + Google Search ───────────────────────────────
    try:
        raw = await call_gemini(api_key, prompt, use_search=True)
        log.info(f"Resposta Gemini: {len(raw)} chars | preview: {raw[:200]}")
        data    = parse_json(raw)
        matches = data.get("matches", [])
        diag["matches_found"] = len(matches)
        log.info(f"{len(matches)} partidas no JSON")
    except json.JSONDecodeError as e:
        err = f"JSON inválido: {str(e)[:100]}"
        log.error(err); diag["errors"].append(err)
        return [], diag
    except Exception as e:
        err = f"Erro ao buscar: {str(e)[:200]}"
        log.error(err); diag["errors"].append(err)
        return [], diag

    # ── Processar cada partida ────────────────────────────────────────────────
    opportunities = []
    for m in matches:
        try:
            name = f"{m.get('homeTeam','?')} vs {m.get('awayTeam','?')}"
            diag["matches_list"].append(name)

            required = ["homeTeam","awayTeam","homeAvgGoalsScored",
                        "homeAvgGoalsConceded","awayAvgGoalsScored","awayAvgGoalsConceded"]
            missing  = [f for f in required if not m.get(f)]
            if missing:
                msg = f"{name}: campos faltando {missing}"
                log.warning(msg); diag["errors"].append(msg); continue

            markets = analyze_match(m, banca, kfrac)
            if not markets:
                diag["ev_results"].append(f"❌ {name}: sem odds válidas"); continue

            best   = markets[0]
            ev_str = f"{name}: EV {best['ev_pct']:+.1f}% ({best['label']} @ {best['odds']:.2f})"
            log.info(f"  {ev_str}")

            if best["ev_pct"] > 0:
                diag["ev_results"].append(f"✅ {ev_str}")
                diag["ev_plus_count"] += 1
            else:
                diag["ev_results"].append(f"➖ {ev_str}"); continue

            # Análise qualitativa
            try:
                opps_txt = "\n".join(
                    f"{i+1}. {mk['label']}: EV {mk['ev_pct']:.1f}% | "
                    f"Edge {mk['edge_pct']:.1f}% | Odd {mk['odds']:.2f}"
                    for i, mk in enumerate(markets[:3])
                )
                ai_txt = await call_gemini(api_key, ANALYSIS_PROMPT.format(
                    home=m["homeTeam"], away=m["awayTeam"],
                    league=m.get("league","?"), date=m.get("date","?"),
                    opps=opps_txt, ctx=m.get("context","N/A"),
                ))
            except Exception:
                ai_txt = "Análise indisponível."

            opportunities.append({
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
            })
        except Exception as e:
            msg = f"Erro em {m.get('homeTeam','?')}: {str(e)[:80]}"
            log.warning(msg); diag["errors"].append(msg)

    opportunities.sort(key=lambda x: x["ev_pct"], reverse=True)
    log.info(f"Scan: {len(opportunities)} EV+ de {diag['matches_found']} partidas")
    return opportunities, diag

# ═══════════════════════════════════════════════════════════════════════════════
# MENSAGENS TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
async def send_opportunity(token, chat_id, opp) -> bool:
    nav     = BETANO_PATH.get(opp["market_key"], ("Principais",opp["market_label"],opp["market_label"]))
    tab, section, option = nav
    retorno = opp["stake"] * opp["odds"]
    lucro   = opp["stake"] * (opp["odds"] - 1)
    odd_min = round(opp["odds"] * 0.97, 2)
    alt     = [m for m in opp.get("all_markets",[])
               if m["ev_pct"] > 0 and m["label"] != opp["market_label"]][:2]
    alt_txt = "\n".join(
        f"   • {m['label']} @ {m['odds']:.2f} — EV {m['ev_pct']:.1f}%" for m in alt
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
        f"→ {tab} → {section}\n"
        f"→ {option}</code>\n\n"
        f"⚠️ Odd mínima: <b>{odd_min}</b>  ⏱ Até 5 min\n\n"
        f"<b>Outros mercados EV+:</b>\n{alt_txt}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧮 Prob real: {opp['real_prob']*100:.1f}%  "
        f"Impl: {opp['implied_prob']*100:.1f}%\n"
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

async def send_diagnostic(token, chat_id, diag, min_ev) -> bool:
    matches_txt = "\n".join(f"  • {m}" for m in diag["matches_list"][:10]) \
                  or "  Nenhuma partida encontrada"
    ev_txt      = "\n".join(f"  {e}" for e in diag["ev_results"][:10]) \
                  or "  Nenhum resultado"
    errors_txt  = "\n".join(f"  ⚠ {e}" for e in diag["errors"][:5]) \
                  if diag["errors"] else ""
    msg = (
        f"🔬 <b>DIAGNÓSTICO DO SCAN</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.now().strftime('%d/%m %H:%M')}\n\n"
        f"📊 Partidas encontradas: <b>{diag['matches_found']}</b>\n"
        f"✅ Com EV positivo: <b>{diag['ev_plus_count']}</b>\n"
        f"📈 EV mínimo: <b>{min_ev:.1f}%</b>\n\n"
        f"<b>Jogos encontrados:</b>\n{matches_txt}\n\n"
        f"<b>EV por jogo:</b>\n{ev_txt}"
    )
    if errors_txt:
        msg += f"\n\n<b>Erros:</b>\n{errors_txt}"
    return await tg_send(token, chat_id, msg)

async def send_startup(token, chat_id, cfg) -> bool:
    msg = (
        f"🟢 <b>BETTING AI ENGINE — ONLINE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🤖 Motor: Google Gemini (gratuito)\n"
        f"💰 Banca: R$ {cfg.banca:.2f}\n"
        f"🛑 Stop loss: R$ {cfg.banca * cfg.stop_loss_pct / 100:.2f}\n"
        f"📈 EV mínimo: {cfg.min_ev_pct}%\n"
        f"🔄 Scan a cada {cfg.scan_interval_min} min\n"
        f"🔬 Diagnóstico: {'ativo' if cfg.debug_mode else 'inativo'}\n"
        f"⚽ Ligas: {', '.join(cfg.leagues)}\n\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    return await tg_send(token, chat_id, msg)

async def send_error(token, chat_id, error) -> bool:
    return await tg_send(token, chat_id,
        f"⚠️ <b>ERRO</b>\n<code>{error[:400]}</code>\n"
        f"🕐 {datetime.now().strftime('%d/%m %H:%M')}")

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
                with open(self.path,"r",encoding="utf-8") as f: return json.load(f)
            except Exception: pass
        return {"tips":[],"perf":{"sent":0,"wins":0,"losses":0}}

    def _save(self):
        try:
            with open(self.path,"w",encoding="utf-8") as f:
                json.dump(self.data,f,ensure_ascii=False,indent=2)
        except Exception as e: log.error(f"Erro ao salvar: {e}")

    def already_sent_today(self, match_key, market_key) -> bool:
        today = date.today().isoformat()
        return any(t["match"]==match_key and t["market"]==market_key
                   and t["date"]==today for t in self.data["tips"])

    def record(self, match_key, opp):
        self.data["tips"].append({
            "match":match_key,"market":opp["market_key"],
            "league":opp["league"],"odds":opp["odds"],
            "stake":opp["stake"],"ev_pct":opp["ev_pct"],
            "date":date.today().isoformat(),"outcome":None,
        })
        self.data["perf"]["sent"] += 1
        if len(self.data["tips"]) > 500:
            self.data["tips"] = self.data["tips"][-500:]
        self._save()

    def get_adaptive_min_ev(self, base):
        p     = self.data["perf"]
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
        opps, diag = await scan(
            cfg.gemini_api_key, cfg.leagues,
            cfg.max_matches, cfg.banca, cfg.kelly_fraction,
        )
        min_ev = memory.get_adaptive_min_ev(cfg.min_ev_pct)

        if cfg.debug_mode:
            await send_diagnostic(cfg.telegram_token, cfg.telegram_chat_id, diag, min_ev)

        if not opps:
            log.info("Sem oportunidades EV+")
            if not cfg.debug_mode:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    f"🔍 Scan {datetime.now().strftime('%d/%m %H:%M')} — Sem EV+")
            return

        filtered = [o for o in opps if o["ev_pct"] >= min_ev]
        log.info(f"{len(filtered)} acima do EV mínimo ({min_ev:.1f}%)")

        sent = 0
        for opp in filtered:
            key = f"{opp['home_team']}-{opp['away_team']}"
            if memory.already_sent_today(key, opp["market_key"]):
                log.info(f"Já enviado hoje: {key}"); continue
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
    log.info("🤖 BETTING AI ENGINE — INICIANDO (Gemini)")
    log.info("=" * 60)

    cfg = Config()
    try:
        cfg.validate()
    except ValueError as e:
        log.error(str(e)); sys.exit(1)

    log.info(f"Banca: R${cfg.banca} | EV mín: {cfg.min_ev_pct}% | "
             f"Scan: {cfg.scan_interval_min}min | Debug: {cfg.debug_mode}")

    memory = Memory()

    ok = await send_startup(cfg.telegram_token, cfg.telegram_chat_id, cfg)
    log.info("✅ Telegram OK" if ok else "❌ Telegram falhou")

    cycle = 0
    while True:
        cycle += 1
        log.info(f"─── Ciclo #{cycle} ───")
        await run_cycle(cfg, memory)
        log.info(f"Aguardando {cfg.scan_interval_min} minutos...")
        await asyncio.sleep(cfg.scan_interval_min * 60)


if __name__ == "__main__":
    asyncio.run(main())
