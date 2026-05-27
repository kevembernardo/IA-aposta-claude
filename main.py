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
        # PL=Premier, PD=La Liga, SA=Serie A, BL1=Bundesliga, FL1=Ligue1
        # BSA=Brasileirao, BSB=Serie B, CPB=Copa Brasil
        # CL=Champions, EL=Europa League, ECL=Conference League
        # CLI=Libertadores, CSA=Sul-Americana
        # DED=Eredivisie, PPL=Liga Portugal, TUR=Super Lig
        # WC=Copa do Mundo, EC=Eurocopa, NL=Nations League
        # MLS=MLS, APD=Argentina, MXN=Liga MX
        self.league_ids           = os.getenv("LEAGUE_IDS", "PL,PD,SA,BL1,FL1,BSA,CL,EL,CLI,CSA").split(",")

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
                    odds: Dict, banca: float, kfrac: float,
                    has_real_data: bool = False) -> List[Dict]:
    lh    = home_avg_scored * away_avg_conceded
    la    = away_avg_scored * home_avg_conceded
    probs = match_probs(lh, la)
    results = []

    # Com dados reais do time permite odds ate 5.00
    # Com medias de liga limita a 3.50 (modelos genericos nao sao confiaveis em azaraos)
    max_odds_allowed = 5.00 if has_real_data else 3.50

    for key, prob in probs.items():
        o = odds.get(key)
        if not o or o <= 1.01: continue

        # Cap de odds por qualidade dos dados
        if o > max_odds_allowed: continue

        # Probabilidade minima — abaixo de 25% o modelo nao e confiavel
        if prob < 0.25: continue

        implied = 1.0 / o

        # Divergencia excessiva: se o mercado ve < 30% e nosso modelo ve > 65%
        # isso e sinal de dado ruim, nao de value bet
        if implied < 0.30 and prob > 0.60: continue

        ev   = calc_ev(prob, o)
        edge = calc_edge(prob, o)

        # Rejeitar EV negativo direto
        if ev <= 0: continue

        # Cap de EV realista:
        # Com dados reais: max 20% (markets eficientes raramente dao mais)
        # Com medias de liga: max 15%
        ev_max = 0.20 if has_real_data else 0.15
        if ev > ev_max: continue

        kf   = calc_kelly(prob, o, kfrac)
        lbl, emoji = classify(ev, edge)
        results.append({
            "key":key,"label":MARKET_LABELS.get(key,key),
            "prob":prob,"implied":implied,"odds":o,
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
    # Europa — Ligas nacionais
    "PL":   "Premier League",
    "PD":   "La Liga",
    "SA":   "Serie A",
    "BL1":  "Bundesliga",
    "FL1":  "Ligue 1",
    "DED":  "Eredivisie",
    "PPL":  "Liga Portugal",
    "PD2":  "Segunda Divisao Espanha",
    "ELC":  "Championship (Ing)",
    "SPL":  "Scottish Premiership",
    "TUR":  "Super Lig Turquia",
    # Europa — Copas
    "CL":   "Champions League",
    "EL":   "Europa League",
    "ECL":  "Conference League",
    "EC":   "Eurocopa",
    "NL":   "Nations League",
    "WC":   "Copa do Mundo",
    # Americas
    "BSA":  "Brasileirao Serie A",
    "BSB":  "Brasileirao Serie B",
    "CPB":  "Copa do Brasil",
    "CLI":  "Libertadores",
    "CSA":  "Sul-Americana",
    "MLS":  "MLS",
    "APD":  "Argentina Primera",
    "MXN":  "Liga MX",
    "COL":  "Liga Colombia",
    "CHI":  "Primera Chile",
    "URU":  "Primera Uruguay",
    # Asia / Outros
    "JPL":  "J-League",
    "CSL":  "Chinese Super League",
}

def fetch_matches_with_ids(api_key: str, league_ids: List[str]) -> List[Dict]:
    """Busca partidas das próximas 48h via football-data.org."""
    today    = date.today()
    in_3_days = today + timedelta(days=2)
    date_from = today.strftime("%Y-%m-%d")
    date_to   = in_3_days.strftime("%Y-%m-%d")

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
                "homeTeam":  home,
                "awayTeam":  away,
                "league":    LEAGUE_NAMES.get(lid, lid),
                "league_id": lid,
                "date":      date_str,
                "id":        m.get("id"),
                "home_id":   m.get("homeTeam", {}).get("id"),
                "away_id":   m.get("awayTeam", {}).get("id"),
            })
        n_found = len(data.get("matches",[]))
        log.info("  %s (%s): %d partidas" % (lid, LEAGUE_NAMES.get(lid,lid), n_found))

    log.info(f"Total de partidas: {len(matches)}")
    return matches


def fetch_team_form(api_key, team_id, n=6):
    url  = 'https://api.football-data.org/v4/teams/%d/matches?limit=%d&status=FINISHED' % (team_id, n)
    data = http_get(url, {'X-Auth-Token': api_key})
    if not data or not data.get('matches'):
        return {'scored':1.3,'conceded':1.3,'form_score':0.5,
                'home_scored':1.4,'home_conceded':1.2,
                'away_scored':1.1,'away_conceded':1.4,'sample':0,'form_str':'-----'}
    matches = data['matches']
    scored_l=[]; conceded_l=[]; home_s=[]; home_c=[]; away_s=[]; away_c=[]; form_pts=[]
    for m in matches:
        score = m.get('score',{}).get('fullTime',{})
        hs = score.get('home',0) or 0
        as_ = score.get('away',0) or 0
        is_home = m.get('homeTeam',{}).get('id') == team_id
        s, c = (hs, as_) if is_home else (as_, hs)
        scored_l.append(s); conceded_l.append(c)
        (home_s if is_home else away_s).append(s)
        (home_c if is_home else away_c).append(c)
        form_pts.append(3 if s > c else (1 if s == c else 0))
    ng = len(scored_l)
    if ng == 0:
        return {'scored':1.3,'conceded':1.3,'form_score':0.5,
                'home_scored':1.4,'home_conceded':1.2,
                'away_scored':1.1,'away_conceded':1.4,'sample':0,'form_str':'-----'}
    # Peso exponencial — jogos recentes pesam mais (0.5^distancia)
    ws = [0.5**(ng-1-i) for i in range(ng)]
    def wavg(lst, w):
        if not lst: return 0.0
        w2=w[:len(lst)]; tw=sum(w2)
        return sum(v*ww for v,ww in zip(lst,w2))/tw if tw>0 else sum(lst)/len(lst)
    form_score = round(wavg(form_pts, ws)/3.0, 3)
    return {
        'scored':        round(wavg(scored_l, ws), 3),
        'conceded':      round(wavg(conceded_l, ws), 3),
        'form_score':    form_score,
        'home_scored':   round(sum(home_s)/max(1,len(home_s)), 3),
        'home_conceded': round(sum(home_c)/max(1,len(home_c)), 3),
        'away_scored':   round(sum(away_s)/max(1,len(away_s)), 3),
        'away_conceded': round(sum(away_c)/max(1,len(away_c)), 3),
        'sample':        ng,
        'form_str':      ''.join('V' if p==3 else 'E' if p==1 else 'D' for p in form_pts[-5:]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# THE ODDS API — odds reais
# ═══════════════════════════════════════════════════════════════════════════════
SPORT_KEYS = {
    # Europa — ligas
    "PL":  "soccer_epl",
    "PD":  "soccer_spain_la_liga",
    "SA":  "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "DED": "soccer_netherlands_eredivisie",
    "PPL": "soccer_portugal_primeira_liga",
    "ELC": "soccer_england_league1",
    "SPL": "soccer_scotland_premiership",
    "TUR": "soccer_turkey_super_league",
    # Europa — copas
    "CL":  "soccer_uefa_champs_league",
    "EL":  "soccer_uefa_europa_league",
    "ECL": "soccer_uefa_europa_conference_league",
    "EC":  "soccer_uefa_european_championship",
    "NL":  "soccer_uefa_nations_league",
    "WC":  "soccer_fifa_world_cup",
    # Americas
    "BSA": "soccer_brazil_campeonato",
    "BSB": "soccer_brazil_serie_b",
    "CPB": "soccer_brazil_copa_do_brasil",
    "CLI": "soccer_conmebol_copa_libertadores",
    "CSA": "soccer_conmebol_copa_sudamericana",
    "MLS": "soccer_usa_mls",
    "APD": "soccer_argentina_primera_division",
    "MXN": "soccer_mexico_ligamx",
    "COL": "soccer_colombia_primera_a",
    # Asia
    "JPL": "soccer_japan_j_league",
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
        import unicodedata
        s = unicodedata.normalize("NFD", s.lower())
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-z0-9]", "", s)

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
        log.warning("GEMINI_API_KEY vazia — análise sem IA")
        return ""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _gemini_sync, api_key, prompt)
        log.info(f"Gemini OK: {len(result)} chars")
        return result
    except Exception as e:
        log.error(f"Gemini ERRO: {e}")
        return ""

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
    log.info('Scan iniciado — %d ligas' % len(cfg.league_ids))
    diag = {'matches_found':0,'with_odds':0,'matches_list':[],
            'ev_results':[],'errors':[],'ev_plus_count':0}
    loop = asyncio.get_event_loop()

    # 1. Buscar partidas
    matches_raw = await loop.run_in_executor(
        _executor, fetch_matches_with_ids, cfg.football_data_key, cfg.league_ids
    )
    diag['matches_found'] = len(matches_raw)
    if not matches_raw:
        diag['errors'].append('Nenhuma partida encontrada')
        return [], diag

    # 2. Buscar odds por liga (paralelo)
    odds_by_league = {}
    for lid in cfg.league_ids:
        sport_key = SPORT_KEYS.get(lid)
        if not sport_key: continue
        odds_list = await loop.run_in_executor(_executor, fetch_odds, cfg.odds_api_key, sport_key)
        odds_by_league[lid] = odds_list

    # 3. Processar cada partida
    opportunities = []
    for m in matches_raw:
        name = '%s vs %s' % (m['homeTeam'], m['awayTeam'])
        diag['matches_list'].append(name)
        lid = m.get('league_id', '')

        # Odds
        odds_list = odds_by_league.get(lid, [])
        odds = match_odds(m['homeTeam'], m['awayTeam'], odds_list)
        if not odds or not any(v for v in odds.values() if v):
            diag['ev_results'].append('❌ %s: sem odds' % name)
            continue
        diag['with_odds'] += 1
        # Limitar odds a 4.50 quando usando medias de liga (sem dados reais)
        # Isso e sobrescrito depois se tiver dados reais do time

        # Stats reais do time (se tiver ID e chave)
        league_avgs = {
            'PL':(1.45,1.10),'PD':(1.35,1.00),'SA':(1.30,1.00),
            'BL1':(1.55,1.15),'FL1':(1.25,0.95),'DED':(1.55,1.20),
            'PPL':(1.35,1.05),'ELC':(1.40,1.10),'SPL':(1.50,1.15),'TUR':(1.45,1.15),
            'CL':(1.50,1.10),'EL':(1.45,1.10),'ECL':(1.40,1.05),
            'EC':(1.25,0.95),'NL':(1.30,1.00),'WC':(1.20,0.90),
            'BSA':(1.40,1.05),'BSB':(1.35,1.05),'CPB':(1.30,1.00),
            'CLI':(1.35,1.00),'CSA':(1.30,1.00),'MLS':(1.50,1.15),
            'APD':(1.45,1.10),'MXN':(1.40,1.05),'COL':(1.35,1.05),
            'CHI':(1.30,1.00),'URU':(1.35,1.05),'JPL':(1.35,1.00),'CSL':(1.40,1.05),
        }
        avg_s, avg_c = league_avgs.get(lid, (1.35, 1.05))
        home_id = m.get('home_id')
        away_id = m.get('away_id')

        # Dados reais por time (quando disponivel)
        home_form = away_form = None
        if cfg.football_data_key and home_id and away_id:
            try:
                home_form = await loop.run_in_executor(_executor, fetch_team_form, cfg.football_data_key, home_id, 6)
                away_form = await loop.run_in_executor(_executor, fetch_team_form, cfg.football_data_key, away_id, 6)
            except Exception as e:
                log.warning('Erro ao buscar form: %s' % e)

        # Calcular lambdas com dados reais ou fallback para media da liga
        if home_form and home_form['sample'] >= 3 and away_form and away_form['sample'] >= 3:
            # Dados reais: casa usa home_scored, fora usa away_scored
            home_attack  = home_form['home_scored']   if home_form['home_scored'] > 0 else home_form['scored']
            home_defense = home_form['home_conceded'] if home_form['home_conceded'] > 0 else home_form['conceded']
            away_attack  = away_form['away_scored']   if away_form['away_scored'] > 0 else away_form['scored']
            away_defense = away_form['away_conceded'] if away_form['away_conceded'] > 0 else away_form['conceded']
            # Ajuste de forma: time em boa forma marca mais / sofre menos
            home_attack  *= (0.85 + 0.30 * home_form['form_score'])
            home_defense *= (1.15 - 0.30 * home_form['form_score'])
            away_attack  *= (0.85 + 0.30 * away_form['form_score'])
            away_defense *= (1.15 - 0.30 * away_form['form_score'])
            data_source = 'real'
            form_str = '%s | %s' % (home_form['form_str'], away_form['form_str'])
        else:
            # Fallback: media da liga + fator campo
            home_attack  = avg_s * 1.10
            home_defense = avg_c * 0.90
            away_attack  = avg_s * 0.90
            away_defense = avg_c * 1.10
            data_source = 'liga'
            form_str = 'N/D'

        markets = analyze_markets(home_attack, home_defense, away_attack, away_defense,
                                  odds, cfg.banca, cfg.kelly_fraction,
                                  has_real_data=(data_source == 'real'))
        if not markets:
            diag['ev_results'].append('❌ %s: sem mercados validos' % name)
            continue

        best   = markets[0]
        ev_str = '%s: EV %+.1f%% (%s @ %.2f) [%s]' % (
            name, best['ev_pct'], best['label'], best['odds'], data_source)
        log.info('  ' + ev_str)

        if best['ev_pct'] > 0:
            diag['ev_results'].append('✅ ' + ev_str)
            diag['ev_plus_count'] += 1
        else:
            diag['ev_results'].append('➖ ' + ev_str)
            continue

        # Analise Gemini — apenas para os top 5 por EV (controlado depois)
        lh = home_attack * away_defense
        la = away_attack * home_defense
        opportunities.append({
            'home_team': m['homeTeam'], 'away_team': m['awayTeam'],
            'league': m['league'], 'date': m['date'],
            'home_form': home_form, 'away_form': away_form,
            'form_str': form_str, 'data_source': data_source,
            'lh': round(lh,3), 'la': round(la,3),
            'market_key': best['key'], 'market_label': best['label'],
            'odds': best['odds'], 'real_prob': best['prob'],
            'implied_prob': best['implied'], 'ev_pct': best['ev_pct'],
            'edge_pct': best['edge_pct'], 'kelly': best['kelly'],
            'stake': best['stake'], 'cls_label': best['cls_label'],
            'cls_color': best['cls_color'], 'all_markets': markets,
            'ai_analysis': '',
        })

    opportunities.sort(key=lambda x: x['ev_pct'], reverse=True)

    # Gemini apenas para top 3 — evita quota 429
    for i, opp in enumerate(opportunities[:3]):
        try:
            opps_txt = '\n'.join(
                '%d. %s: EV %.1f%% | Edge %.1f%% | Odd %.2f' %
                (j+1, mk['label'], mk['ev_pct'], mk['edge_pct'], mk['odds'])
                for j, mk in enumerate(opp['all_markets'][:3])
            )
            hf = opp.get('home_form') or {}
            af = opp.get('away_form') or {}
            form_context = ''
            if hf.get('sample',0) >= 3:
                form_context = ('Forma: %s(casa) gols %.1f/%.1f | %s(fora) gols %.1f/%.1f' % (
                    opp['home_team'], hf.get('scored',0), hf.get('conceded',0),
                    opp['away_team'], af.get('scored',0), af.get('conceded',0)))
            prompt = (
                'Analista quantitativo de apostas. Tecnico e direto.\n'
                '%s vs %s — %s — %s\n'
                'Lambda casa: %.3f gols/jogo | Lambda fora: %.3f gols/jogo\n'
                'Fonte dos dados: %s\n'
                '%s\n'
                'Forma recente: %s\n\n'
                'Mercados EV+:\n%s\n\n'
                '3 partes (max 130 palavras):\n'
                '1. VANTAGEM MATEMATICA\n'
                '2. RISCO PRINCIPAL\n'
                '3. ENTRADA RECOMENDADA (mercado + odd minima)'
            ) % (
                opp['home_team'], opp['away_team'], opp['league'], opp['date'],
                opp['lh'], opp['la'], opp['data_source'], form_context,
                opp['form_str'], opps_txt,
            )
            opp['ai_analysis'] = await call_gemini(cfg.gemini_api_key, prompt)
            if i < 2: await asyncio.sleep(5)
        except Exception as e:
            log.warning('Gemini erro: %s' % e)
            opp['ai_analysis'] = ''

    log.info('Scan: %d EV+ de %d partidas (%d com odds)' % (
        len(opportunities), diag['matches_found'], diag['with_odds']))
    return opportunities, diag

class Memory:
    def __init__(self, path='data/history.json'):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path,'r',encoding='utf-8') as f: return json.load(f)
            except Exception: pass
        return {'tips':[],'perf':{'sent':0,'wins':0,'losses':0,'staked':0.0,'returned':0.0},'banca':None,'silent_until':None}

    def _save(self):
        try:
            with open(self.path,'w',encoding='utf-8') as f:
                json.dump(self.data,f,ensure_ascii=False,indent=2)
        except Exception as e: log.error('Erro ao salvar: %s' % e)

    # ── BANCA ─────────────────────────────────────────────────────────────────
    def get_banca(self, cfg_banca): return self.data.get('banca') or cfg_banca

    def set_banca(self, valor):
        self.data['banca'] = round(float(valor), 2)
        self._save()
        log.info('Banca atualizada: R$ %.2f' % valor)

    def registrar_resultado(self, ganhou, valor):
        banca = self.data.get('banca', 0) or 0
        p = self.data['perf']
        if ganhou:
            p['wins']     += 1
            p['returned']  = p.get('returned',0) + valor
            nova = round(banca + valor, 2)
        else:
            p['losses']   += 1
            p['staked']    = p.get('staked',0) + valor
            nova = round(max(0, banca - valor), 2)
        self.data['banca'] = nova
        self._save()
        return nova

    # ── MODO SILENCIOSO ───────────────────────────────────────────────────────
    def is_silent(self):
        until = self.data.get('silent_until')
        if not until: return False
        from datetime import datetime as _dt
        try:
            return _dt.now() < _dt.fromisoformat(until)
        except Exception: return False

    def set_silent(self, hours):
        from datetime import datetime as _dt, timedelta as _td
        until = (_dt.now() + _td(hours=hours)).isoformat()
        self.data['silent_until'] = until
        self._save()
        return until

    def clear_silent(self):
        self.data['silent_until'] = None
        self._save()

    # ── HISTORICO ─────────────────────────────────────────────────────────────
    def already_sent_today(self, key) -> bool:
        today = date.today().isoformat()
        return any(t['match']==key and t['date']==today for t in self.data['tips'])

    def record(self, key, opp):
        self.data['tips'].append({
            'match':key,'market':opp.get('market_key',''),
            'league':opp.get('league',''),'odds':opp.get('odds',0),
            'stake':opp.get('stake',0),'ev_pct':opp.get('ev_pct',0),
            'date':date.today().isoformat(),'outcome':None,
            'home_team':opp.get('home_team',''),'away_team':opp.get('away_team',''),
        })
        self.data['perf']['sent'] += 1
        if len(self.data['tips']) > 1000:
            self.data['tips'] = self.data['tips'][-1000:]
        self._save()

    def get_pending_results(self, days_back=2):
        from datetime import datetime as _dt, timedelta as _td
        cutoff = (_dt.now() - _td(days=days_back)).date().isoformat()
        return [
            t for t in self.data['tips']
            if t.get('outcome') is None
            and t.get('market','') != 'strategy'
            and t.get('date','') >= cutoff
            and t.get('home_team','')
        ]

    def update_outcome(self, tip_id_or_match, outcome, returned=0.0):
        for t in self.data['tips']:
            if t.get('match') == tip_id_or_match and t.get('outcome') is None:
                t['outcome']  = outcome
                t['returned'] = returned
                p = self.data['perf']
                if outcome == 'win':
                    p['wins']     += 1
                    p['returned']  = p.get('returned',0) + returned
                    banca = self.data.get('banca',0) or 0
                    self.data['banca'] = round(banca + returned - t.get('stake',0), 2)
                elif outcome == 'loss':
                    p['losses']  += 1
                    p['staked']   = p.get('staked',0) + t.get('stake',0)
                self._save()
                return True
        return False

    # ── ESTATISTICAS ──────────────────────────────────────────────────────────
    def get_stats(self):
        p = self.data['perf']
        total = p['wins'] + p['losses']
        wr    = (p['wins']/total*100) if total > 0 else 0
        staked   = p.get('staked',0)
        returned = p.get('returned',0)
        roi = ((returned-staked)/staked*100) if staked > 0 else 0
        return {
            'banca':    self.data.get('banca',0),
            'enviados': p['sent'], 'wins': p['wins'], 'losses': p['losses'],
            'total': total, 'win_rate': round(wr,1), 'roi': round(roi,1),
            'staked': round(staked,2), 'returned': round(returned,2),
        }

    def get_weekly_stats(self):
        from datetime import datetime as _dt, timedelta as _td
        week_ago = (_dt.now() - _td(days=7)).date().isoformat()
        tips = [t for t in self.data['tips'] if t.get('date','') >= week_ago and t.get('market','') != 'strategy']
        wins    = sum(1 for t in tips if t.get('outcome')=='win')
        losses  = sum(1 for t in tips if t.get('outcome')=='loss')
        staked  = sum(t.get('stake',0) for t in tips if t.get('outcome') in ('win','loss'))
        returned= sum(t.get('returned',0) for t in tips if t.get('outcome')=='win')
        roi     = ((returned-staked)/staked*100) if staked > 0 else 0
        best    = max((t for t in tips if t.get('outcome')=='win'), key=lambda x: x.get('returned',0), default=None)
        worst   = max((t for t in tips if t.get('outcome')=='loss'), key=lambda x: x.get('stake',0), default=None)
        by_league = {}
        for t in tips:
            lg = t.get('league','?')
            if lg not in by_league: by_league[lg] = {'w':0,'l':0}
            if t.get('outcome')=='win': by_league[lg]['w'] += 1
            elif t.get('outcome')=='loss': by_league[lg]['l'] += 1
        return {
            'total': len(tips), 'wins': wins, 'losses': losses,
            'pending': len(tips)-wins-losses,
            'staked': round(staked,2), 'returned': round(returned,2),
            'roi': round(roi,1),
            'win_rate': round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0,
            'best': best, 'worst': worst,
            'by_league': by_league,
        }

    def needs_weekly_report(self):
        today = date.today()
        if today.weekday() != 6: return False  # so domingo
        key = 'weekly-report-%s' % today.isoformat()
        return not self.already_sent_today(key)

    def get_adaptive_min_ev(self, base):
        p = self.data['perf']
        total = p['wins'] + p['losses']
        if total < 20: return base
        wr = p['wins'] / total
        if wr > 0.58: return max(base*0.85, 2.0)
        if wr < 0.38: return base*1.25
        return base


from itertools import combinations as _comb

def _combined_odd(sels):
    r = 1.0
    for o in sels: r *= o["odds"]
    return round(r, 2)

def _combined_prob(sels):
    r = 1.0
    for o in sels: r *= o["real_prob"]
    return r

def _combined_ev(sels):
    p = _combined_prob(sels)
    od = _combined_odd(sels)
    return calc_ev(p, od)


# ═══════════════════════════════════════════════════════════════════════════════
# MODO DE BANCA
# ═══════════════════════════════════════════════════════════════════════════════
def get_banca_mode(banca):
    if banca <= 50:
        return {
            "nome": "SOBREVIVENCIA", "emoji": "🔴",
            "descricao": "Banca crítica — foco em preservação e alavancagem segura",
            "kelly_frac": 0.08, "max_apostas": 1,
            "min_prob": 0.48, "min_odds": 1.40, "max_odds": 5.00,
            "min_ev": 4.0, "dupla_ok": True,
            "dupla_min_prob": 0.58, "dupla_min_ev": 0.05, "kelly_dupla": 0.10,
            "tripla_ok": False, "yankee_ok": False, "canadian_ok": False,
            "aposta_min": 1.00,
            "objetivo": "Dobrar a banca antes de qualquer outra estrategia",
        }
    elif banca <= 200:
        return {
            "nome": "RECUPERACAO", "emoji": "🟡",
            "descricao": "Banca baixa — crescimento gradual com risco controlado",
            "kelly_frac": 0.12, "max_apostas": 2,
            "min_prob": 0.50, "min_odds": 1.40, "max_odds": 2.80,
            "min_ev": 3.0, "dupla_ok": True,
            "dupla_min_prob": 0.52, "dupla_min_ev": 0.03, "kelly_dupla": 0.12,
            "tripla_ok": False, "yankee_ok": False, "canadian_ok": False,
            "aposta_min": 1.00,
            "objetivo": "Atingir R$200 com disciplina e entradas selecionadas",
        }
    elif banca <= 500:
        return {
            "nome": "CRESCIMENTO", "emoji": "🟢",
            "descricao": "Banca media — estrategia equilibrada",
            "kelly_frac": 0.18, "max_apostas": 3,
            "min_prob": 0.46, "min_odds": 1.30, "max_odds": 3.50,
            "min_ev": 2.0, "dupla_ok": True,
            "dupla_min_prob": 0.48, "dupla_min_ev": 0.02, "kelly_dupla": 0.15,
            "tripla_ok": True, "yankee_ok": False, "canadian_ok": False,
            "aposta_min": 1.00,
            "objetivo": "Crescimento consistente rumo a R$500+",
        }
    else:
        return {
            "nome": "NORMAL", "emoji": "🔵",
            "descricao": "Banca saudavel — sistema completo ativo",
            "kelly_frac": 0.25, "max_apostas": 99,
            "min_prob": 0.44, "min_odds": 1.20, "max_odds": 5.00,
            "min_ev": 1.0, "dupla_ok": True,
            "dupla_min_prob": 0.46, "dupla_min_ev": 0.02, "kelly_dupla": 0.15,
            "tripla_ok": True, "yankee_ok": True, "canadian_ok": True,
            "aposta_min": 1.00,
            "objetivo": "Maximizar crescimento com diversificacao total",
        }


def filter_by_mode(opps, mode, banca):
    """
    Filtra oportunidades pelo modo de banca.
    - min_prob: probabilidade minima calculada pelo Poisson
    - max_odds: apenas para ESTRATEGIA (multiplas) — alertas individuais aceitam ate 5.00
    - min_ev: EV minimo configurado
    """
    result = []
    for o in opps:
        # Filtro de probabilidade minima — garante qualidade do modelo
        if o["real_prob"] < mode["min_prob"]: continue
        # Filtro de odds para alerta individual: 1.30 ate 5.00
        # (acima de 5.00 o modelo com medias de liga nao e confiavel)
        if not (mode["min_odds"] <= o["odds"] <= 5.00): continue
        # Filtro de EV minimo
        if o["ev_pct"] < mode["min_ev"]: continue
        kf    = mode["kelly_frac"]
        stake = banca * calc_kelly(o["real_prob"], o["odds"], kf)
        stake = max(mode["aposta_min"], round(stake, 2))
        stake = min(stake, banca * 0.20)
        o2 = dict(o)
        o2["stake"] = stake
        result.append(o2)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE ESTRATEGIA
# ═══════════════════════════════════════════════════════════════════════════════
from itertools import combinations as _comb

def _combined_odd(sels):
    r = 1.0
    for o in sels: r *= o["odds"]
    return round(r, 2)

def _combined_prob(sels):
    r = 1.0
    for o in sels: r *= o["real_prob"]
    return r

def _combined_ev(sels):
    return calc_ev(_combined_prob(sels), _combined_odd(sels))


def assess_day(opps, banca, min_ev_cfg):
    mode      = get_banca_mode(banca)
    filtered  = filter_by_mode(opps, mode, banca)
    n         = len(filtered)
    good      = filtered[:mode["max_apostas"]]
    strong    = [o for o in filtered if o["ev_pct"] >= mode["min_ev"] + 2]
    leagues   = [o["league"] for o in good]
    n_leagues = len(set(leagues))
    diversified = n_leagues >= max(1, len(good) // 2)
    base = {"n": n, "strong": len(strong), "n_leagues": n_leagues, "mode": mode, "banca": banca}

    if n == 0:
        return dict(base, rec="AGUARDAR", type=None, sels=[],
            reason=("Nenhuma entrada passa os criterios do modo %s. Prob min: %.0f%%, Odds: %.2f-%.2f, EV min: %.1f%%."
                    % (mode["nome"], mode["min_prob"]*100, mode["min_odds"], mode["max_odds"], mode["min_ev"])),
            action="Nao apostar. Monitorando proximas rodadas.",
            risk="NULO", color="⬜")

    if mode["nome"] == "SOBREVIVENCIA":
        best = good[0]
        if len(good) >= 2 and mode["dupla_ok"]:
            o1, o2 = good[0], good[1]
            if o1["real_prob"] >= mode["dupla_min_prob"] and o2["real_prob"] >= mode["dupla_min_prob"]:
                ev2 = _combined_ev([o1, o2])
                od2 = _combined_odd([o1, o2])
                p2  = _combined_prob([o1, o2])
                if ev2 >= mode["dupla_min_ev"]:
                    stake2 = max(mode["aposta_min"], round(banca * calc_kelly(p2, od2, mode["kelly_dupla"]), 2))
                    stake2 = min(stake2, banca * 0.35)
                    ret2   = round(stake2 * od2, 2)
                    return dict(base, rec="DUPLA ALAVANCAGEM", type="double", sels=[o1, o2],
                        comb_odd=od2, comb_ev=round(ev2*100,1), comb_prob=round(p2*100,1),
                        stake=stake2, ret=ret2,
                        reason=("Banca critica (R$ %.2f). Dupla de alavancagem: ambas prob >= %.0f%% e EV combinado +%.1f%%. Retorno R$ %.2f."
                                % (banca, mode["dupla_min_prob"]*100, ev2*100, ret2)),
                        action=("Aposte R$ %.2f na dupla (odd %.2f). STOP LOSS: R$ %.2f."
                                % (stake2, od2, banca * 0.50)),
                        risk="CONTROLADO", color="🟡")
        stake1 = max(mode["aposta_min"], round(banca * calc_kelly(best["real_prob"], best["odds"], mode["kelly_frac"]), 2))
        stake1 = min(stake1, banca * 0.25)
        return dict(base, rec="SIMPLES CIRURGICA", type="single", sels=[best],
            reason=("Banca critica (R$ %.2f). 1 entrada precisa: prob %.1f%%, EV +%.1f%%."
                    % (banca, best["real_prob"]*100, best["ev_pct"])),
            action=("Aposte R$ %.2f em %s vs %s — %s @ %.2f. Stop loss: R$ %.2f."
                    % (stake1, best["home_team"], best["away_team"], best["market_label"], best["odds"], banca*0.50)),
            risk="BAIXO-CONTROLADO", color="🟢")

    if mode["nome"] == "RECUPERACAO":
        if len(good) == 1:
            o = good[0]
            return dict(base, rec="SIMPLES", type="single", sels=[o],
                reason=("1 oportunidade valida. EV +%.1f%%, prob %.1f%%."
                        % (o["ev_pct"], o["real_prob"]*100)),
                action=("Aposte R$ %.2f em %s vs %s — %s @ %.2f."
                        % (o["stake"], o["home_team"], o["away_team"], o["market_label"], o["odds"])),
                risk="BAIXO", color="🟢")
        o1, o2 = good[0], good[1]
        ev2 = _combined_ev([o1, o2])
        od2 = _combined_odd([o1, o2])
        p2  = _combined_prob([o1, o2])
        if ev2 >= mode["dupla_min_ev"] and o1["real_prob"] >= mode["dupla_min_prob"] and diversified:
            stake2 = max(mode["aposta_min"], round(banca * calc_kelly(p2, od2, mode["kelly_dupla"]), 2))
            stake2 = min(stake2, banca * 0.20)
            return dict(base, rec="DUPLA + 2 SIMPLES", type="double_plus", sels=[o1, o2],
                comb_odd=od2, comb_ev=round(ev2*100,1), comb_prob=round(p2*100,1),
                stake=stake2, ret=round(stake2*od2,2),
                total_alt=round(o1["stake"]+o2["stake"],2),
                reason=("2 oportunidades EV combinado +%.1f%%. Dupla ou 2 simples separadas." % (ev2*100)),
                action=("OPCAO A — Dupla: R$ %.2f (odd %.2f, retorno R$ %.2f).\nOPCAO B — Simples: R$ %.2f + R$ %.2f."
                        % (stake2, od2, round(stake2*od2,2), o1["stake"], o2["stake"])),
                risk="MODERADO", color="🟡")
        return dict(base, rec="SIMPLES SEPARADAS", type="singles", sels=good[:2],
            reason=("2 oportunidades. Dupla nao justificada (EV %.1f%%)." % (ev2*100)),
            action=("Aposte R$ %.2f e R$ %.2f separadamente." % (good[0]["stake"], good[1]["stake"])),
            total_stake=round(sum(o["stake"] for o in good[:2]),2),
            risk="BAIXO", color="🟢")

    if mode["nome"] == "CRESCIMENTO":
        if len(good) >= 3 and len(strong) >= 2:
            top3 = good[:3]
            ev3 = _combined_ev(top3); od3 = _combined_odd(top3); p3 = _combined_prob(top3)
            if ev3 > 0.04 and diversified:
                stake3 = max(1.0, round(banca * calc_kelly(p3, od3, 0.08), 2))
                return dict(base, rec="TRIPLA INTELIGENTE", type="treble", sels=top3,
                    comb_odd=od3, comb_ev=round(ev3*100,1), comb_prob=round(p3*100,1),
                    stake=stake3, ret=round(stake3*od3,2),
                    reason=("3 selecoes EV combinado +%.1f%% em %d ligas." % (ev3*100, n_leagues)),
                    action=("R$ %.2f na tripla (odd %.2f, retorno R$ %.2f)." % (stake3, od3, round(stake3*od3,2))),
                    risk="MODERADO", color="🟡")
        if len(good) >= 2:
            o1, o2 = good[0], good[1]
            ev2 = _combined_ev([o1,o2]); od2 = _combined_odd([o1,o2]); p2 = _combined_prob([o1,o2])
            if ev2 > 0.02:
                stake2 = max(1.0, round(banca * calc_kelly(p2, od2, mode["kelly_dupla"]), 2))
                return dict(base, rec="DUPLA", type="double", sels=[o1,o2],
                    comb_odd=od2, comb_ev=round(ev2*100,1), comb_prob=round(p2*100,1),
                    stake=stake2, ret=round(stake2*od2,2),
                    reason=("2 oportunidades solidas. EV combinado +%.1f%%." % (ev2*100)),
                    action=("R$ %.2f na dupla (odd %.2f)." % (stake2, od2)),
                    risk="MODERADO", color="🟡")
        return dict(base, rec="SIMPLES SEPARADAS", type="singles", sels=good,
            reason=("%d oportunidades. Kelly %.0f%% por entrada." % (n, mode["kelly_frac"]*100)),
            action="Aposte cada uma separadamente.",
            total_stake=round(sum(o["stake"] for o in good),2),
            risk="BAIXO", color="🟢")

    # Modo NORMAL
    if len(good) >= 5 and len(strong) >= 4:
        top5 = good[:5]; unit5 = round(banca*0.004,2)
        best2 = max(_combined_ev(list(c))*100 for c in _comb(top5,2))
        return dict(base, rec="CANADIAN", type="canadian", sels=top5,
            count=26, unit=unit5, total_stake=round(unit5*26,2),
            max_ret=round(unit5*_combined_odd(top5),2), best_ev2=round(best2,1),
            reason=("%d selecoes fortes em %d ligas." % (n, n_leagues)),
            action=("26 apostas x R$ %.2f = R$ %.2f total." % (unit5, unit5*26)),
            risk="ALTO", color="🔴")
    if len(good) >= 4 and len(strong) >= 3:
        top4 = good[:4]; unit = round(banca*0.005,2)
        min_od = min(_combined_odd(list(c)) for c in _comb(top4,2))
        best2  = max(_combined_ev(list(c))*100 for c in _comb(top4,2))
        return dict(base, rec="YANKEE", type="yankee", sels=top4,
            count=11, unit=unit, total_stake=round(unit*11,2),
            max_ret=round(unit*_combined_odd(top4),2), min_od=round(min_od,2), best_ev2=round(best2,1),
            reason=("%d selecoes fortes. Basta 2/4 (odd min %.2f)." % (len(strong), min_od)),
            action=("11 apostas x R$ %.2f = R$ %.2f total." % (unit, unit*11)),
            risk="MODERADO-ALTO", color="🟠")
    if len(good) >= 3:
        top3 = good[:3]
        ev3 = _combined_ev(top3); od3 = _combined_odd(top3); p3 = _combined_prob(top3)
        if ev3 > 0.03:
            stake3 = max(1.0, round(banca * calc_kelly(p3, od3, 0.10), 2))
            return dict(base, rec="TRIPLA", type="treble", sels=top3,
                comb_odd=od3, comb_ev=round(ev3*100,1), comb_prob=round(p3*100,1),
                stake=stake3, ret=round(stake3*od3,2),
                reason=("3 selecoes EV combinado +%.1f%%." % (ev3*100)),
                action=("R$ %.2f na tripla (odd %.2f)." % (stake3, od3)),
                risk="MODERADO", color="🟡")
    if len(good) >= 2:
        o1, o2 = good[0], good[1]
        ev2 = _combined_ev([o1,o2]); od2 = _combined_odd([o1,o2]); p2 = _combined_prob([o1,o2])
        if ev2 > 0.02:
            stake2 = max(1.0, round(banca * calc_kelly(p2, od2, 0.15), 2))
            return dict(base, rec="DUPLA", type="double", sels=[o1,o2],
                comb_odd=od2, comb_ev=round(ev2*100,1), comb_prob=round(p2*100,1),
                stake=stake2, ret=round(stake2*od2,2),
                reason=("2 selecoes EV combinado +%.1f%%." % (ev2*100)),
                action=("R$ %.2f na dupla (odd %.2f)." % (stake2, od2)),
                risk="MODERADO", color="🟡")
    return dict(base, rec="SIMPLES SEPARADAS", type="singles", sels=good,
        reason=("%d oportunidades. Kelly %.0f%%." % (n, mode["kelly_frac"]*100)),
        action="Aposte cada uma separadamente.",
        total_stake=round(sum(o["stake"] for o in good),2),
        risk="BAIXO", color="🟢")


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIO DA ESTRATEGIA
# ═══════════════════════════════════════════════════════════════════════════════
async def send_strategy(token, chat_id, strat, banca, gemini_key):
    mode  = strat.get("mode", {})
    rec   = strat["rec"]
    color = strat["color"]
    risk  = strat["risk"]
    sels  = strat.get("sels", [])
    n     = strat["n"]

    mode_line = "%s Modo: <b>%s</b> — %s" % (mode.get("emoji",""), mode.get("nome",""), mode.get("descricao",""))

    sel_lines = []
    for i, o in enumerate(sels):
        sel_lines.append(
            "  %d. %s vs %s\n     %s @ %.2f | EV +%.1f%% | Prob %.1f%%" %
            (i+1, o["home_team"], o["away_team"], o["market_label"], o["odds"], o["ev_pct"], o["real_prob"]*100))
    sels_txt = "\n".join(sel_lines) if sel_lines else "  Nenhuma selecao hoje"

    t = strat.get("type")
    extra_lines = []
    if t in ("double", "double_plus", "treble"):
        extra_lines = [
            "📊 Odd combinada: <b>%.2f</b>" % strat.get("comb_odd", 0),
            "📈 EV combinado: <b>+%.1f%%</b>" % strat.get("comb_ev", 0),
            "🎯 Prob de acerto: <b>%.1f%%</b>" % strat.get("comb_prob", 0),
            "💰 Apostar: <b>R$ %.2f</b>" % strat.get("stake", 0),
            "💵 Retorno se acertar: <b>R$ %.2f</b>" % strat.get("ret", 0),
        ]
        if t == "double_plus":
            extra_lines.append("↔️ Alt (2 simples): R$ %.2f total" % strat.get("total_alt", 0))
    elif t == "yankee":
        extra_lines = [
            "📋 11 apostas x R$ %.2f = <b>R$ %.2f total</b>" % (strat.get("unit",0), strat.get("total_stake",0)),
            "🛡 Ret min (2/4): odd %.2f" % strat.get("min_od", 0),
            "🏆 Ret max (4/4): R$ %.2f" % strat.get("max_ret", 0),
        ]
    elif t == "canadian":
        extra_lines = [
            "📋 26 apostas x R$ %.2f = <b>R$ %.2f total</b>" % (strat.get("unit",0), strat.get("total_stake",0)),
            "🏆 Ret max: R$ %.2f" % strat.get("max_ret", 0),
        ]
    elif t in ("singles", "single"):
        ts = strat.get("total_stake", sum(o.get("stake",0) for o in sels))
        extra_lines = ["💰 Total: <b>R$ %.2f</b>" % ts]
    extra_txt = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

    msg = (
        "📋 <b>ESTRATEGIA DO DIA</b> — %s\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "%s\n"
        "💰 Banca atual: <b>R$ %.2f</b>\n"
        "🎯 Objetivo: %s\n\n"
        "%s <b>RECOMENDACAO: %s</b>\n"
        "⚖️ Risco: %s\n"
        "📊 %d validas | %d fortes\n\n"
        "<b>Selecoes:</b>\n%s\n"
        "%s\n\n"
        "<b>Por que:</b>\n%s\n\n"
        "<b>Como executar:</b>\n%s"
    ) % (
        datetime.now().strftime("%d/%m/%Y"),
        mode_line, banca, mode.get("objetivo",""),
        color, rec, risk, n, strat["strong"],
        sels_txt, extra_txt,
        strat.get("reason",""), strat.get("action",""),
    )
    ok = await tg_send(token, chat_id, msg)
    await asyncio.sleep(2)

    if sels and gemini_key:
        sel_sum = "; ".join(
            "%s vs %s %s @%.2f EV+%.1f%% Prob%.1f%%" %
            (o["home_team"], o["away_team"], o["market_label"], o["odds"], o["ev_pct"], o["real_prob"]*100)
            for o in sels)
        prompt = (
            "Analista profissional de apostas. Direto e tecnico.\n"
            "SITUACAO: Banca R$ %.2f | Modo %s (%s) | Objetivo: %s\n"
            "ESTRATEGIA: %s | Risco: %s\n"
            "SELECOES: %s\n"
            "MOTIVO: %s\n\n"
            "Analise em 4 partes (max 200 palavras):\n"
            "1. SITUACAO DE BANCA: avalie o contexto com R$ %.2f\n"
            "2. VALIDACAO: concorda com a estrategia?\n"
            "3. RISCO REAL: o que pode dar errado hoje\n"
            "4. CONSELHO FINAL: instrucao clara e direta"
        ) % (banca, mode.get("nome",""), mode.get("descricao",""), mode.get("objetivo",""),
             rec, risk, sel_sum, strat.get("reason",""), banca)
        ai_txt = await call_gemini(gemini_key, prompt)
        if ai_txt:
            await tg_send(token, chat_id,
                "🧠 <b>ANALISE DA IA — ESTRATEGIA</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n\n%s" % ai_txt[:3500])
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# HANDLER DE COMANDOS DO TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
def tg_get_updates_sync(token, offset=0):
    url = "https://api.telegram.org/bot%s/getUpdates?offset=%d&timeout=5" % (token, offset)
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("result", []) if data.get("ok") else []
    except Exception:
        return []


async def send_weekly_report(token, chat_id, memory, gemini_key):
    stats = memory.get_weekly_stats()
    w = stats
    by_league_txt = ""
    for lg, s in sorted(w["by_league"].items(), key=lambda x: x[1]["w"]+x[1]["l"], reverse=True)[:5]:
        wr = round(s["w"]/(s["w"]+s["l"])*100,0) if (s["w"]+s["l"])>0 else 0
        by_league_txt += "  %s: %dV %dD (%.0f%%)\n" % (lg, s["w"], s["l"], wr)

    best_txt  = ("%s vs %s +R$%.2f" % (w["best"]["home_team"],  w["best"]["away_team"],  w["best"].get("returned",0)))  if w.get("best")  else "—"
    worst_txt = ("%s vs %s -R$%.2f" % (w["worst"]["home_team"], w["worst"]["away_team"], w["worst"].get("stake",0)))    if w.get("worst") else "—"

    msg = (
        "📊 <b>RELATORIO SEMANAL</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "📅 Semana encerrada em %s\n\n"
        "📤 Sinais enviados: <b>%d</b>\n"
        "✅ Vitorias: <b>%d</b>\n"
        "❌ Derrotas: <b>%d</b>\n"
        "⏳ Pendentes: <b>%d</b>\n"
        "🎯 Win rate: <b>%.1f%%</b>\n"
        "💰 Total apostado: R$ %.2f\n"
        "💵 Total retornado: R$ %.2f\n"
        "📈 ROI da semana: <b>%+.1f%%</b>\n\n"
        "<b>Por liga:</b>\n%s\n"
        "<b>Melhor entrada:</b> %s\n"
        "<b>Pior entrada:</b> %s\n\n"
        "Sistema continua monitorando. 💪"
    ) % (
        date.today().strftime("%d/%m/%Y"),
        w["total"], w["wins"], w["losses"], w["pending"],
        w["win_rate"], w["staked"], w["returned"], w["roi"],
        by_league_txt or "  Sem dados\n",
        best_txt, worst_txt,
    )
    ok = await tg_send(token, chat_id, msg)

    # Analise Gemini do desempenho semanal
    if gemini_key and w["total"] > 0:
        prompt = (
            "Analista de apostas. Avalie o desempenho semanal:\n"
            "Sinais: %d | Vitorias: %d | Derrotas: %d | Win rate: %.1f%% | ROI: %+.1f%%\n"
            "Apostado: R$ %.2f | Retornado: R$ %.2f\n"
            "Responda em 3 partes (max 150 palavras):\n"
            "1. AVALIACAO: como foi a semana matematicamente\n"
            "2. PADROES: o que os resultados indicam\n"
            "3. AJUSTE: o que mudar na proxima semana"
        ) % (w["total"], w["wins"], w["losses"], w["win_rate"], w["roi"], w["staked"], w["returned"])
        ai = await call_gemini(gemini_key, prompt)
        if ai:
            await tg_send(token, chat_id,
                "🧠 <b>ANALISE SEMANAL DA IA</b>\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n%s" % ai[:3000])
    return ok


async def process_commands(cfg, memory):
    """Verifica mensagens no Telegram e processa comandos."""
    loop = asyncio.get_event_loop()
    offset_file = "data/tg_offset.json"
    offset = 0
    if os.path.exists(offset_file):
        try:
            offset = json.load(open(offset_file)).get("offset", 0)
        except Exception:
            pass

    updates = await loop.run_in_executor(_executor, tg_get_updates_sync, cfg.telegram_token, offset)

    for upd in updates:
        offset = upd["update_id"] + 1
        msg    = upd.get("message", {})
        text   = msg.get("text", "").strip()
        chat   = str(msg.get("chat", {}).get("id", ""))

        # Aceitar apenas do chat configurado
        if chat != cfg.telegram_chat_id:
            continue

        parts = text.split()
        cmd   = parts[0].lower() if parts else ""

        if cmd == "/banca":
            if len(parts) < 2:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "Uso: /banca 150.00\nExemplo: /banca 47.50")
                continue
            try:
                valor = float(parts[1].replace(",", "."))
                memory.set_banca(valor)
                cfg.banca = valor
                mode = get_banca_mode(valor)
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "✅ <b>Banca atualizada!</b>\n"
                    "💰 Nova banca: <b>R$ %.2f</b>\n"
                    "%s Modo ativo: <b>%s</b>\n"
                    "🎯 %s" % (valor, mode["emoji"], mode["nome"], mode["objetivo"]))
            except ValueError:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "Valor invalido. Use: /banca 150.00")

        elif cmd == "/ganhou":
            if len(parts) < 2:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "Uso: /ganhou 47.20\n(informe o LUCRO obtido)")
                continue
            try:
                lucro = float(parts[1].replace(",", "."))
                nova  = memory.registrar_resultado(True, lucro)
                cfg.banca = nova
                mode  = get_banca_mode(nova)
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "✅ <b>Vitoria registrada!</b>\n"
                    "💵 Lucro: +R$ %.2f\n"
                    "💰 Banca atual: <b>R$ %.2f</b>\n"
                    "%s Modo: <b>%s</b>" % (lucro, nova, mode["emoji"], mode["nome"]))
            except ValueError:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Valor invalido.")

        elif cmd == "/perdeu":
            if len(parts) < 2:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "Uso: /perdeu 13.08\n(informe o valor que apostou)")
                continue
            try:
                perda = float(parts[1].replace(",", "."))
                nova  = memory.registrar_resultado(False, perda)
                cfg.banca = nova
                mode  = get_banca_mode(nova)
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "❌ <b>Derrota registrada.</b>\n"
                    "💸 Perda: -R$ %.2f\n"
                    "💰 Banca atual: <b>R$ %.2f</b>\n"
                    "%s Modo: <b>%s</b>" % (perda, nova, mode["emoji"], mode["nome"]))
            except ValueError:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Valor invalido.")

        elif cmd == "/status":
            stats = memory.get_stats()
            mode  = get_banca_mode(stats["banca"])
            today_tips = [t for t in memory.data["tips"] if t["date"] == date.today().isoformat()]
            today_str  = "\n".join(
                "  • %s vs %s — %s @ %.2f" % (
                    t["match"].split("-")[0], t["match"].split("-")[1],
                    t.get("market",""), t.get("odds",0))
                for t in today_tips[-5:]
            ) or "  Nenhum sinal hoje ainda"
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                "📊 <b>STATUS DO SISTEMA</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "%s Modo: <b>%s</b>\n"
                "💰 Banca: <b>R$ %.2f</b>\n"
                "🎯 Objetivo: %s\n\n"
                "<b>Historico:</b>\n"
                "📤 Sinais enviados: %d\n"
                "✅ Vitorias: %d\n"
                "❌ Derrotas: %d\n"
                "🎯 Win rate: %.1f%%\n"
                "📈 ROI: %+.1f%%\n\n"
                "<b>Sinais de hoje:</b>\n%s\n\n"
                "<i>Use /banca 150 para atualizar\n"
                "/ganhou 47.20 para registrar vitoria\n"
                "/perdeu 13.08 para registrar derrota</i>" % (
                mode["emoji"], mode["nome"],
                stats["banca"], mode["objetivo"],
                stats["enviados"], stats["wins"], stats["losses"],
                stats["win_rate"], stats["roi"],
                today_str))

        elif cmd == "/silencio":
            horas = int(parts[1]) if len(parts) > 1 else 8
            until = memory.set_silent(horas)
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                "🔇 <b>Modo silencioso ativado por %d horas.</b>\n"
                "Scans continuam mas alertas pausados ate %s.\n"
                "Use /ativar para reativar." % (horas, until[:16]))

        elif cmd == "/ativar":
            memory.clear_silent()
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                "🔊 <b>Alertas reativados!</b> Sistema voltou ao normal.")

        elif cmd == "/relatorio":
            await send_weekly_report(cfg.telegram_token, cfg.telegram_chat_id, memory, cfg.gemini_api_key)

        elif cmd == "/ligas":
            ligas_txt = "\n".join("  %s — %s" % (k,v) for k,v in sorted(LEAGUE_NAMES.items()))
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                "⚽ <b>LIGAS SUPORTADAS</b>\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n%s\n\n"
                "Para ativar, va em Railway → Variables → LEAGUE_IDS\n"
                "Exemplo: PL,PD,BSA,CLI,CSA" % ligas_txt)

        elif cmd == "/ajuda":
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                "🤖 <b>COMANDOS DISPONIVEIS</b>\n\n"
                "/banca 150.00 — atualiza a banca\n"
                "/ganhou 47.20 — registra vitoria (lucro)\n"
                "/perdeu 13.08 — registra derrota (valor apostado)\n"
                "/status — banca, modo e historico\n"
                "/silencio 8 — silenciar por 8 horas\n"
                "/ativar — reativar alertas\n"
                "/relatorio — relatorio semanal\n"
                "/ligas — ligas disponiveis\n"
                "/ajuda — esta mensagem")

    # Salvar offset
    try:
        json.dump({"offset": offset}, open(offset_file, "w"))
    except Exception:
        pass


async def run_cycle(cfg: Config, memory: Memory):
    log.info("=" * 50)
    log.info("Ciclo — %s" % datetime.now().strftime("%d/%m/%Y %H:%M"))
    try:
        cfg.banca = memory.get_banca(cfg.banca)
        mode = get_banca_mode(cfg.banca)
        log.info("Modo: %s (R$ %.2f)" % (mode["nome"], cfg.banca))

        # Modo silencioso ativo — pular alertas mas continuar scans
        if memory.is_silent():
            log.info("Modo silencioso ativo — pulando alertas")
            return

        # Relatorio semanal (domingo)
        if memory.needs_weekly_report():
            await send_weekly_report(cfg.telegram_token, cfg.telegram_chat_id, memory, cfg.gemini_api_key)
            wr_key = "weekly-report-%s" % date.today().isoformat()
            memory.record(wr_key, {"market_key":"weekly","league":"system","odds":1.0,"stake":0,"ev_pct":0})

        opps, diag = await scan(cfg)
        min_ev = memory.get_adaptive_min_ev(cfg.min_ev_pct)

        if cfg.debug_mode:
            await send_diagnostic(cfg.telegram_token, cfg.telegram_chat_id, diag, min_ev)

        if not opps:
            log.info("Sem oportunidades EV+ apos filtros de sanidade")
            return

        mode_filtered = filter_by_mode(opps, mode, cfg.banca)
        filtered      = [o for o in mode_filtered if o["ev_pct"] >= min_ev]
        log.info("%d oportunidades passaram todos os filtros" % len(filtered))

        if not filtered:
            log.info("Nenhuma oportunidade valida hoje (modo %s, EV min %.1f%%)" % (mode["nome"], min_ev))
            # Avisar o usuario por que nao ha entradas
            msg_no_filter = ("🔍 Scan %s — %d jogos EV+ encontrados, nenhum passou os filtros do modo %s. "
                             "Criterios: prob >= %.0f%%, odds %.1f-5.0, EV >= %.1f%%. Monitorando." % (
                             datetime.now().strftime("%d/%m %H:%M"),
                             len(opps), mode["nome"], mode["min_prob"]*100, mode["min_odds"], min_ev))
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id, msg_no_filter)
            return

        # Enviar alertas individuais
        sent = 0
        max_ind = mode["max_apostas"]
        for opp in filtered[:max_ind]:
            key = "%s-%s-%s" % (opp["home_team"], opp["away_team"], opp["market_key"])
            if memory.already_sent_today(key):
                log.info("Ja enviado hoje: %s" % key)
                continue
            ok = await send_opportunity(cfg.telegram_token, cfg.telegram_chat_id, opp)
            if ok:
                await asyncio.sleep(1)
                if opp.get("ai_analysis"):
                    await send_analysis(cfg.telegram_token, cfg.telegram_chat_id, opp)
                memory.record(key, opp)
                sent += 1
                await asyncio.sleep(2)
        log.info("%d alertas individuais enviados" % sent)

        # Estrategia do dia (1x por dia)
        strategy_key = "strategy-%s" % date.today().isoformat()
        if not memory.already_sent_today(strategy_key):
            log.info("Montando estrategia do dia (modo %s)..." % mode["nome"])
            strat = assess_day(filtered, cfg.banca, min_ev)
            await asyncio.sleep(3)
            await send_strategy(cfg.telegram_token, cfg.telegram_chat_id,
                                strat, cfg.banca, cfg.gemini_api_key)
            memory.record(strategy_key, {
                "market_key":"strategy","league":"system","odds":1.0,"stake":0,"ev_pct":0,
                "home_team":"","away_team":""
            })
            log.info("Estrategia enviada: %s" % strat["rec"])

    except Exception as e:
        log.error("Erro no ciclo: %s" % e, exc_info=True)
        try: await send_error(cfg.telegram_token, cfg.telegram_chat_id, str(e))
        except Exception: pass


async def main():
    log.info("=" * 60)
    log.info("Iniciando Betting AI Engine")
    log.info("=" * 60)
    cfg = Config()
    try:
        cfg.validate()
    except ValueError as e:
        log.error(str(e)); sys.exit(1)

    memory = Memory()
    # Carregar banca da memoria se disponivel
    cfg.banca = memory.get_banca(cfg.banca)
    mode = get_banca_mode(cfg.banca)
    log.info("Banca: R$ %.2f | Modo: %s | EV min: %.1f%% | Scan: %dmin" % (
        cfg.banca, mode["nome"], cfg.min_ev_pct, cfg.scan_interval_min))

    ok = await send_startup(cfg.telegram_token, cfg.telegram_chat_id, cfg)
    log.info("Telegram OK" if ok else "Telegram falhou")
    await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
        "💡 <b>Comandos disponiveis:</b>\n"
        "/banca 150 — atualizar banca\n"
        "/ganhou 47.20 — registrar vitoria\n"
        "/perdeu 13.08 — registrar derrota\n"
        "/status — ver banca e historico\n"
        "/ajuda — todos os comandos")

    cycle = 0
    while True:
        cycle += 1
        log.info("--- Ciclo #%d ---" % cycle)
        # Verificar comandos antes de cada ciclo
        await process_commands(cfg, memory)
        await run_cycle(cfg, memory)
        log.info("Aguardando %d minutos..." % cfg.scan_interval_min)
        # Verificar comandos periodicamente enquanto espera
        for _ in range(cfg.scan_interval_min):
            await asyncio.sleep(60)
            await process_commands(cfg, memory)


if __name__ == "__main__":
    asyncio.run(main())
