"""
BETTING AI ENGINE — Sistema Autônomo 24/7
Stack 100% gratuita:
  - football-data.org  → partidas e estatísticas (grátis)
  - The Odds API       → odds reais (500 req/mês grátis)
  - Google Gemini      → análise qualitativa (grátis, sem search)
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
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional

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

_executor = ThreadPoolExecutor(max_workers=3)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    def __init__(self):
        self.gemini_api_key       = os.getenv("GEMINI_API_KEY", "")
        self.football_data_key    = os.getenv("FOOTBALL_DATA_KEY", "")
        self.odds_api_key         = os.getenv("ODDS_API_KEY", "")
        self.telegram_token       = os.getenv("TELEGRAM_TOKEN", "")
        self.telegram_chat_id     = os.getenv("TELEGRAM_CHAT_ID", "")
        self.banca                = float(os.getenv("BANCA", "1000"))
        self.stop_loss_pct        = float(os.getenv("STOP_LOSS_PCT", "5"))
        self.kelly_fraction       = float(os.getenv("KELLY_FRACTION", "0.25"))
        self.min_ev_pct           = float(os.getenv("MIN_EV_PCT", "1"))
        self.scan_interval_min    = int(os.getenv("SCAN_INTERVAL_MIN", "180"))
        self.debug_mode           = os.getenv("DEBUG_MODE", "true").lower() == "true"
        # IDs das ligas no football-data.org:
        # PL=39(Premier), PD=140(La Liga), SA=135(Serie A),
        # BL1=78(Bundesliga), FL1=61(Ligue1), BSA=71(Brasileirao), CL=2(Champions)
        self.league_ids           = os.getenv("LEAGUE_IDS", "PL,PD,SA,BSA,CL").split(",")

    def validate(self):
        errors = []
        if not self.telegram_token:    errors.append("TELEGRAM_TOKEN não definida")
        if not self.telegram_chat_id:  errors.append("TELEGRAM_CHAT_ID não definida")
        if not self.football_data_key: errors.append("FOOTBALL_DATA_KEY não definida")
        if not self.odds_api_key:      errors.append("ODDS_API_KEY não definida")
        if errors:
            raise ValueError("Configuração inválida:\n" + "\n".join(f"  • {e}" for e in errors))

# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR MATEMÁTICO — POISSON + KELLY + EV
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

def analyze_markets(home_avg_scored, home_avg_conceded,
                    away_avg_scored, away_avg_conceded,
                    odds: Dict, banca: float, kfrac: float) -> List[Dict]:
    lh    = home_avg_scored * away_avg_conceded
    la    = away_avg_scored * home_avg_conceded
    probs = match_probs(lh, la)
    results = []
    for key, prob in probs.items():
        o = odds.get(key)
        if not o or o <= 1.01: continue
        ev   = calc_ev(prob, o)
        edge = calc_edge(prob, o)
        kf   = calc_kelly(prob, o, kfrac)
        lbl, emoji = classify(ev, edge)
        results.append({
            "key":key,"label":MARKET_LABELS.get(key,key),
            "prob":prob,"implied":1/o,"odds":o,
            "ev":ev,"ev_pct":ev*100,"edge_pct":edge,
            "kelly":kf,"stake":banca*kf,
            "cls_label":lbl,"cls_color":emoji,
        })
    return sorted(results, key=lambda x: x["ev_pct"], reverse=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP HELPER — urllib stdlib
# ═══════════════════════════════════════════════════════════════════════════════
def http_get(url: str, headers: Dict = None) -> Optional[Dict]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        log.error(f"HTTP {e.code} em {url}: {e.read().decode()[:200]}")
        return None
    except Exception as e:
        log.error(f"Erro em {url}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTBALL-DATA.ORG — partidas e estatísticas
# ═══════════════════════════════════════════════════════════════════════════════
LEAGUE_NAMES = {
    "PL":"Premier League","PD":"La Liga","SA":"Serie A",
    "BL1":"Bundesliga","FL1":"Ligue 1","BSA":"Brasileirão","CL":"Champions League",
}

def fetch_matches(api_key: str, league_ids: List[str]) -> List[Dict]:
    """Busca partidas das próximas 48h via football-data.org."""
    today    = date.today()
    tomorrow = today + timedelta(days=1)
    date_from = today.strftime("%Y-%m-%d")
    date_to   = tomorrow.strftime("%Y-%m-%d")

    headers  = {"X-Auth-Token": api_key}
    matches  = []

    for lid in league_ids:
        url  = (f"https://api.football-data.org/v4/competitions/{lid}/matches"
                f"?dateFrom={date_from}&dateTo={date_to}&status=SCHEDULED")
        data = http_get(url, headers)
        if not data:
            continue
        for m in data.get("matches", []):
            home = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name","?")
            away = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name","?")
            utc  = m.get("utcDate","")
            try:
                dt = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
                # Converter para horário de Brasília (UTC-3)
                dt_br = dt - timedelta(hours=3)
                date_str = dt_br.strftime("%d/%m %H:%M")
            except Exception:
                date_str = utc[:16]

            matches.append({
                "homeTeam": home,
                "awayTeam": away,
                "league":   LEAGUE_NAMES.get(lid, lid),
                "league_id": lid,
                "date":     date_str,
                "id":       m.get("id"),
            })
        log.info(f"  {lid}: {len(data.get('matches',[]))} partidas")

    log.info(f"Total de partidas: {len(matches)}")
    return matches

def fetch_team_stats(api_key: str, team_id: int, competition_id: str) -> Dict:
    """Busca médias de gols dos últimos jogos."""
    url  = f"https://api.football-data.org/v4/teams/{team_id}/matches?limit=5&status=FINISHED"
    data = http_get(url, {"X-Auth-Token": api_key})
    if not data:
        return {"scored": 1.3, "conceded": 1.3}

    scored = conceded = count = 0
    for m in data.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        home_s = score.get("home", 0) or 0
        away_s = score.get("away", 0) or 0
        if m.get("homeTeam", {}).get("id") == team_id:
            scored   += home_s
            conceded += away_s
        else:
            scored   += away_s
            conceded += home_s
        count += 1

    if count == 0:
        return {"scored": 1.3, "conceded": 1.3}
    return {"scored": round(scored / count, 2), "conceded": round(conceded / count, 2)}

# ═══════════════════════════════════════════════════════════════════════════════
# THE ODDS API — odds reais
# ═══════════════════════════════════════════════════════════════════════════════
SPORT_KEYS = {
    "PL":  "soccer_england_league1",
    "PD":  "soccer_spain_la_liga",
    "SA":  "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "BSA": "soccer_brazil_campeonato",
    "CL":  "soccer_uefa_champs_league",
}

def fetch_odds(api_key: str, sport_key: str) -> List[Dict]:
    """Busca odds via The Odds API — 500 req/mês grátis."""
    url  = (f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            f"?apiKey={api_key}&regions=eu&markets=h2h,totals"
            f"&oddsFormat=decimal&dateFormat=iso")
    data = http_get(url)
    if not data or not isinstance(data, list):
        return []

    result = []
    for game in data:
        home = game.get("home_team","")
        away = game.get("away_team","")
        odds = {"home": None, "draw": None, "away": None,
                "over25": None, "under25": None}

        for bm in game.get("bookmakers", [])[:3]:
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == home and not odds["home"]:
                            odds["home"] = o["price"]
                        elif o["name"] == away and not odds["away"]:
                            odds["away"] = o["price"]
                        elif o["name"] == "Draw" and not odds["draw"]:
                            odds["draw"] = o["price"]
                elif mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        pt = o.get("point", 0)
                        if abs(pt - 2.5) < 0.1:
                            if o["name"] == "Over" and not odds["over25"]:
                                odds["over25"] = o["price"]
                            elif o["name"] == "Under" and not odds["under25"]:
                                odds["under25"] = o["price"]

        result.append({"home": home, "away": away, "odds": odds})

    log.info(f"  {sport_key}: {len(result)} jogos com odds")
    return result

def match_odds(match_name_home: str, match_name_away: str,
               odds_list: List[Dict]) -> Optional[Dict]:
    """Casa as odds com o jogo pelo nome do time."""
    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    nh = normalize(match_name_home)
    na = normalize(match_name_away)

    best, best_score = None, 0
    for o in odds_list:
        oh = normalize(o["home"])
        oa = normalize(o["away"])
        # Match exato ou parcial
        score = 0
        if nh in oh or oh in nh: score += 2
        if na in oa or oa in na: score += 2
        if nh[:4] in oh: score += 1
        if na[:4] in oa: score += 1
        if score > best_score:
            best_score = score
            best = o

    return best["odds"] if best and best_score >= 3 else None

# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI — análise qualitativa (sem search, gratuito)
# ═══════════════════════════════════════════════════════════════════════════════
def _gemini_sync(api_key: str, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp  = model.generate_content(prompt)
    return resp.text or ""

async def call_gemini(api_key: str, prompt: str) -> str:
    if not api_key:
        return "Análise indisponível (GEMINI_API_KEY não configurada)."
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_executor, _gemini_sync, api_key, prompt)
    except Exception as e:
        log.warning(f"Gemini erro: {e}")
        return "Análise indisponível."

ANALYSIS_PROMPT = """Analista de apostas esportivas. Objetivo e técnico.

{home} vs {away} — {league} — {date}
λ Casa: {lh:.2f} gols/jogo | λ Fora: {la:.2f} gols/jogo

Mercados com valor matemático (EV positivo):
{opps}

Responda em 3 partes curtas (máx 120 palavras):
1. VANTAGEM: por que o EV é positivo aqui
2. RISCO: principal fator que pode invalidar
3. ENTRADA: mercado exato e odd mínima"""

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
def tg_send_sync(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        log.error("Telegram não configurado"); return False
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id, "text": text[:4096],
        "parse_mode": "HTML", "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
        headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode()).get("ok", False)
    except urllib.error.HTTPError as e:
        log.error(f"Telegram {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        log.error(f"Telegram erro: {e}"); return False

async def tg_send(token, chat_id, text) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, tg_send_sync, token, chat_id, text)

async def send_opportunity(token, chat_id, opp) -> bool:
    nav     = BETANO_PATH.get(opp["market_key"], ("Principais",opp["market_label"],opp["market_label"]))
    tab, section, option = nav
    retorno = opp["stake"] * opp["odds"]
    lucro   = opp["stake"] * (opp["odds"] - 1)
    odd_min = round(opp["odds"] * 0.97, 2)
    alt = [m for m in opp.get("all_markets",[])
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
    matches_txt = "\n".join(f"  • {m}" for m in diag["matches_list"][:12]) \
                  or "  Nenhuma partida encontrada"
    ev_txt = "\n".join(f"  {e}" for e in diag["ev_results"][:12]) \
             or "  Nenhum resultado"
    errors_txt = "\n".join(f"  ⚠ {e}" for e in diag["errors"][:4]) \
                 if diag["errors"] else ""
    msg = (
        f"🔬 <b>DIAGNÓSTICO DO SCAN</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.now().strftime('%d/%m %H:%M')}\n\n"
        f"📊 Partidas: <b>{diag['matches_found']}</b>  "
        f"Com odds: <b>{diag['with_odds']}</b>\n"
        f"✅ EV positivo: <b>{diag['ev_plus_count']}</b>  "
        f"EV mínimo: <b>{min_ev:.1f}%</b>\n\n"
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
        f"📡 Dados: football-data.org + The Odds API\n"
        f"🤖 Análise: Google Gemini\n"
        f"💰 Banca: R$ {cfg.banca:.2f}\n"
        f"🛑 Stop loss: R$ {cfg.banca * cfg.stop_loss_pct / 100:.2f}\n"
        f"📈 EV mínimo: {cfg.min_ev_pct}%\n"
        f"🔄 Scan a cada {cfg.scan_interval_min} min\n"
        f"⚽ Ligas: {', '.join(cfg.league_ids)}\n\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    return await tg_send(token, chat_id, msg)

async def send_error(token, chat_id, error) -> bool:
    return await tg_send(token, chat_id,
        f"⚠️ <b>ERRO</b>\n<code>{str(error)[:400]}</code>\n"
        f"🕐 {datetime.now().strftime('%d/%m %H:%M')}")

# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
async def scan(cfg: Config) -> Tuple[List[Dict], Dict]:
    log.info(f"Scan iniciado — ligas: {cfg.league_ids}")
    diag = {"matches_found":0,"with_odds":0,"matches_list":[],
            "ev_results":[],"errors":[],"ev_plus_count":0}

    # ── 1. Buscar partidas ────────────────────────────────────────────────────
    loop    = asyncio.get_event_loop()
    matches = await loop.run_in_executor(
        _executor, fetch_matches, cfg.football_data_key, cfg.league_ids
    )
    diag["matches_found"] = len(matches)
    if not matches:
        diag["errors"].append("Nenhuma partida encontrada no football-data.org")
        return [], diag

    # ── 2. Buscar odds por liga ───────────────────────────────────────────────
    odds_by_league: Dict[str, List[Dict]] = {}
    for lid in cfg.league_ids:
        sport_key = SPORT_KEYS.get(lid)
        if not sport_key:
            continue
        odds_list = await loop.run_in_executor(
            _executor, fetch_odds, cfg.odds_api_key, sport_key
        )
        odds_by_league[lid] = odds_list

    # ── 3. Cruzar partidas com odds e calcular EV ─────────────────────────────
    opportunities = []
    for m in matches:
        name = f"{m['homeTeam']} vs {m['awayTeam']}"
        diag["matches_list"].append(name)

        # Buscar odds
        lid       = m.get("league_id","")
        odds_list = odds_by_league.get(lid, [])
        odds      = match_odds(m["homeTeam"], m["awayTeam"], odds_list)

        if not odds or not any(v for v in odds.values() if v):
            diag["ev_results"].append(f"❌ {name}: sem odds")
            continue
        diag["with_odds"] += 1

        # Usar médias padrão por liga (suficiente para Poisson)
        # Valores médios reais de cada liga:
        league_avgs = {
            "PL":  (1.45, 1.10), "PD":  (1.35, 1.00), "SA":  (1.30, 1.00),
            "BL1": (1.55, 1.15), "FL1": (1.25, 0.95), "BSA": (1.40, 1.05),
            "CL":  (1.50, 1.10),
        }
        avg_s, avg_c = league_avgs.get(lid, (1.35, 1.05))

        # Casa marca mais, fora concede mais (fator campo)
        home_scored   = avg_s * 1.10
        home_conceded = avg_c * 0.90
        away_scored   = avg_s * 0.90
        away_conceded = avg_c * 1.10

        markets = analyze_markets(
            home_scored, home_conceded,
            away_scored, away_conceded,
            odds, cfg.banca, cfg.kelly_fraction
        )
        if not markets:
            diag["ev_results"].append(f"❌ {name}: sem mercados válidos")
            continue

        best   = markets[0]
        ev_str = f"{name}: EV {best['ev_pct']:+.1f}% ({best['label']} @ {best['odds']:.2f})"
        log.info(f"  {ev_str}")

        if best["ev_pct"] > 0:
            diag["ev_results"].append(f"✅ {ev_str}")
            diag["ev_plus_count"] += 1
        else:
            diag["ev_results"].append(f"➖ {ev_str}")
            continue

        # Análise qualitativa via Gemini
        lh = home_scored * away_conceded
        la = away_scored * home_conceded
        try:
            opps_txt = "\n".join(
                f"{i+1}. {mk['label']}: EV {mk['ev_pct']:.1f}% | "
                f"Edge {mk['edge_pct']:.1f}% | Odd {mk['odds']:.2f}"
                for i, mk in enumerate(markets[:3])
            )
            ai_txt = await call_gemini(cfg.gemini_api_key, ANALYSIS_PROMPT.format(
                home=m["homeTeam"], away=m["awayTeam"],
                league=m["league"], date=m["date"],
                lh=lh, la=la, opps=opps_txt,
            ))
        except Exception:
            ai_txt = "Análise indisponível."

        opportunities.append({
            "home_team":m["homeTeam"],"away_team":m["awayTeam"],
            "league":m["league"],"date":m["date"],
            "market_key":best["key"],"market_label":best["label"],
            "odds":best["odds"],"real_prob":best["prob"],
            "implied_prob":best["implied"],"ev_pct":best["ev_pct"],
            "edge_pct":best["edge_pct"],"kelly":best["kelly"],
            "stake":best["stake"],"cls_label":best["cls_label"],
            "cls_color":best["cls_color"],"all_markets":markets,
            "ai_analysis":ai_txt,
        })

    opportunities.sort(key=lambda x: x["ev_pct"], reverse=True)
    log.info(f"Scan: {len(opportunities)} EV+ de {diag['matches_found']} partidas")
    return opportunities, diag

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

    def already_sent_today(self, key) -> bool:
        today = date.today().isoformat()
        return any(t["match"]==key and t["date"]==today for t in self.data["tips"])

    def record(self, key, opp):
        self.data["tips"].append({
            "match":key,"market":opp["market_key"],
            "league":opp["league"],"odds":opp["odds"],
            "stake":opp["stake"],"ev_pct":opp["ev_pct"],
            "date":date.today().isoformat(),"outcome":None,
        })
        self.data["perf"]["sent"] += 1
        if len(self.data["tips"]) > 500:
            self.data["tips"] = self.data["tips"][-500:]
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
        opps, diag = await scan(cfg)
        min_ev     = memory.get_adaptive_min_ev(cfg.min_ev_pct)

        if cfg.debug_mode:
            await send_diagnostic(cfg.telegram_token, cfg.telegram_chat_id, diag, min_ev)

        if not opps:
            log.info("Sem oportunidades EV+")
            return

        filtered = [o for o in opps if o["ev_pct"] >= min_ev]
        log.info(f"{len(filtered)} acima do EV mínimo ({min_ev:.1f}%)")

        sent = 0
        for opp in filtered:
            key = f"{opp['home_team']}-{opp['away_team']}-{opp['market_key']}"
            if memory.already_sent_today(key):
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
    log.info("🤖 BETTING AI ENGINE — INICIANDO")
    log.info("=" * 60)

    cfg = Config()
    try:
        cfg.validate()
    except ValueError as e:
        log.error(str(e)); sys.exit(1)

    log.info(f"Banca: R${cfg.banca} | EV mín: {cfg.min_ev_pct}% | "
             f"Scan: {cfg.scan_interval_min}min")

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
