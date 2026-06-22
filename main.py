"""
BETTING AI ENGINE — FAVORITOS DE ALTA PROBABILIDADE
Estrutura simplificada: busca entradas com odd proxima de 1.50
e alta probabilidade de acerto, em qualquer liga configurada.

Stack 100% gratuita:
  - football-data.org  -> partidas e estatisticas
  - The Odds API       -> odds reais (h2h, totals, btts, handicap asiatico)
  - Google Gemini      -> analise qualitativa
  - Telegram           -> alertas
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


# ============================================================================
# CONFIGURACAO
# ============================================================================
class Config:
    def __init__(self):
        self.gemini_api_key    = os.getenv("GEMINI_API_KEY", "")
        self.football_data_key = os.getenv("FOOTBALL_DATA_KEY", "")
        self.odds_api_key      = os.getenv("ODDS_API_KEY", "")
        self.telegram_token    = os.getenv("TELEGRAM_TOKEN", "")
        self.telegram_chat_id  = os.getenv("TELEGRAM_CHAT_ID", "")

        self.banca             = float(os.getenv("BANCA", "1000"))
        self.kelly_fraction    = float(os.getenv("KELLY_FRACTION", "0.15"))
        self.stake_cap_pct     = float(os.getenv("STAKE_CAP_PCT", "5"))      # % max da banca por entrada

        # Alvo de odd ~1.50
        self.odd_min           = float(os.getenv("ODD_MIN", "1.35"))
        self.odd_max           = float(os.getenv("ODD_MAX", "1.65"))
        self.min_prob          = float(os.getenv("MIN_PROB", "0.68"))        # 68%+ de confianca
        self.max_entries       = int(os.getenv("MAX_ENTRIES", "5"))          # entradas por scan

        self.scan_interval_min = int(os.getenv("SCAN_INTERVAL_MIN", "120"))
        self.debug_mode        = os.getenv("DEBUG_MODE", "true").lower() == "true"

        self.league_ids        = os.getenv(
            "LEAGUE_IDS",
            "PL,PD,SA,BL1,FL1,DED,PPL,ELC,SPL,TUR,BEL,SUI,AUT,GRE,POL,DEN,NOR,SWE,"
            "BSA,BSB,CLI,CSA,MLS,APD,MXN,COL,CHI,JPL,KOR,CSL,AUS,WC,EC,CL,EL"
        ).split(",")

    def validate(self):
        errors = []
        if not self.telegram_token:    errors.append("TELEGRAM_TOKEN nao definida")
        if not self.telegram_chat_id:  errors.append("TELEGRAM_CHAT_ID nao definida")
        if not self.football_data_key: errors.append("FOOTBALL_DATA_KEY nao definida")
        if not self.odds_api_key:      errors.append("ODDS_API_KEY nao definida")
        if errors:
            raise ValueError("Configuracao invalida:\n" + "\n".join("  - %s" % e for e in errors))


# ============================================================================
# MOTOR MATEMATICO — POISSON + EV + KELLY
# ============================================================================
def poisson_prob(lam: float, k: int) -> float:
    if lam <= 0: return 1.0 if k == 0 else 0.0
    log_p = -lam + k * math.log(lam)
    for i in range(1, k + 1): log_p -= math.log(i)
    return math.exp(log_p)


def match_probs(lh: float, la: float, max_g: int = 8, h1_ratio: float = 0.42) -> Dict:
    """
    Calcula probabilidades de TODOS os mercados a partir da distribuicao de Poisson:
      - Resultado (home/draw/away)
      - Over/Under em varias linhas: 0.5, 1.5, 2.5, 3.5
      - Ambas marcam
      - Handicap asiatico (+-0.5)
      - Over/Under no PRIMEIRO TEMPO (0.5 e 1.5), estimado como uma fracao
        dos gols esperados do jogo todo (h1_ratio ~42% e a media historica
        de gols que acontecem na primeira etapa nas principais ligas).
    """
    home = draw = away = btts = 0.0
    over_lines = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0, 3.5: 0.0}

    for h in range(max_g + 1):
        ph = poisson_prob(lh, h)
        for a in range(max_g + 1):
            p = ph * poisson_prob(la, a)
            total = h + a
            if h > a:           home += p
            elif h == a:        draw += p
            else:               away += p
            if h > 0 and a > 0: btts += p
            for line in over_lines:
                if total > line:
                    over_lines[line] += p

    result = {
        "home": home, "draw": draw, "away": away, "btts": btts,
        "ah_home_m5": home,        "ah_home_p5": home + draw,
        "ah_away_m5": away,        "ah_away_p5": away + draw,
    }
    for line, p_over in over_lines.items():
        key = str(line).replace(".", "")  # 0.5 -> "05", 2.5 -> "25"
        result["over%s" % key]  = p_over
        result["under%s" % key] = 1 - p_over

    # Primeiro tempo — total de gols esperado e uma fracao do jogo inteiro
    lambda_h1_total = (lh + la) * h1_ratio
    over_h1 = {0.5: 0.0, 1.5: 0.0}
    max_h1 = 6
    for t in range(max_h1 + 1):
        p = poisson_prob(lambda_h1_total, t)
        for line in over_h1:
            if t > line:
                over_h1[line] += p
    for line, p_over in over_h1.items():
        key = str(line).replace(".", "")
        result["h1_over%s" % key]  = p_over
        result["h1_under%s" % key] = 1 - p_over

    return result


def calc_ev(prob, odds):   return prob * (odds - 1) - (1 - prob)
def calc_edge(prob, odds): return (prob - 1 / odds) * 100


def calc_kelly(prob, odds, frac=0.15):
    b = odds - 1
    if b <= 0: return 0.0
    k = (b * prob - (1 - prob)) / b
    return max(0.0, k * frac)


MARKET_LABELS = {
    "home": "Vitoria Casa", "draw": "Empate", "away": "Vitoria Fora",
    "over05": "Over 0.5", "under05": "Under 0.5",
    "over15": "Over 1.5", "under15": "Under 1.5",
    "over25": "Over 2.5", "under25": "Under 2.5",
    "over35": "Over 3.5", "under35": "Under 3.5",
    "h1_over05": "Over 0.5 (1T)", "h1_under05": "Under 0.5 (1T)",
    "h1_over15": "Over 1.5 (1T)", "h1_under15": "Under 1.5 (1T)",
    "btts": "Ambas Marcam",
    "ah_home_m5": "Handicap Asiatico Casa -0.5", "ah_home_p5": "Handicap Asiatico Casa +0.5",
    "ah_away_m5": "Handicap Asiatico Fora -0.5", "ah_away_p5": "Handicap Asiatico Fora +0.5",
}

BETANO_PATH = {
    "home":       ("Principais", "Resultado Final", "1 - Vitoria Casa"),
    "draw":       ("Principais", "Resultado Final", "X - Empate"),
    "away":       ("Principais", "Resultado Final", "2 - Vitoria Visitante"),
    "over05":     ("Gols", "Total de Gols", "Mais de 0.5"),
    "under05":    ("Gols", "Total de Gols", "Menos de 0.5"),
    "over15":     ("Gols", "Total de Gols", "Mais de 1.5"),
    "under15":    ("Gols", "Total de Gols", "Menos de 1.5"),
    "over25":     ("Gols", "Total de Gols", "Mais de 2.5"),
    "under25":    ("Gols", "Total de Gols", "Menos de 2.5"),
    "over35":     ("Gols", "Total de Gols", "Mais de 3.5"),
    "under35":    ("Gols", "Total de Gols", "Menos de 3.5"),
    "h1_over05":  ("1o Tempo", "Total de Gols 1o Tempo", "Mais de 0.5"),
    "h1_under05": ("1o Tempo", "Total de Gols 1o Tempo", "Menos de 0.5"),
    "h1_over15":  ("1o Tempo", "Total de Gols 1o Tempo", "Mais de 1.5"),
    "h1_under15": ("1o Tempo", "Total de Gols 1o Tempo", "Menos de 1.5"),
    "btts":       ("Gols", "Ambas Equipes Marcam", "Sim"),
    "ah_home_m5": ("Handicap", "Handicap Asiatico", "Casa -0.5"),
    "ah_home_p5": ("Handicap", "Handicap Asiatico", "Casa +0.5"),
    "ah_away_m5": ("Handicap", "Handicap Asiatico", "Fora -0.5"),
    "ah_away_p5": ("Handicap", "Handicap Asiatico", "Fora +0.5"),
}


def find_favorite_markets(home_attack, home_defense, away_attack, away_defense,
                           odds: Dict, cfg: Config) -> List[Dict]:
    """
    Varre TODOS os mercados do jogo e retorna apenas os que tem
    odd dentro da faixa alvo (~1.50) E alta probabilidade real.
    Nao filtra por EV — o criterio e exclusivamente odd + confianca.
    """
    lh = home_attack * away_defense
    la = away_attack * home_defense
    probs = match_probs(lh, la)

    results = []
    for key, prob in probs.items():
        o = odds.get(key)
        if not o or o <= 1.01:
            continue

        # Filtro 1: odd precisa estar na faixa alvo (~1.50)
        if not (cfg.odd_min <= o <= cfg.odd_max):
            continue

        # Filtro 2: probabilidade real precisa ser alta
        if prob < cfg.min_prob:
            continue

        ev   = calc_ev(prob, o)
        edge = calc_edge(prob, o)
        kf   = calc_kelly(prob, o, cfg.kelly_fraction)
        stake = cfg.banca * kf
        stake = min(stake, cfg.banca * cfg.stake_cap_pct / 100.0)
        stake = max(1.0, round(stake, 2))

        # Distancia ao alvo 1.50 — usado para desempate no ranking
        dist_to_target = abs(o - 1.50)

        results.append({
            "key": key, "label": MARKET_LABELS.get(key, key),
            "prob": prob, "implied": 1.0 / o, "odds": o,
            "ev": ev, "ev_pct": ev * 100, "edge_pct": edge,
            "kelly": kf, "stake": stake,
            "dist_to_target": dist_to_target,
        })

    # Ranking: maior probabilidade primeiro, depois menor distancia da odd alvo
    results.sort(key=lambda x: (-x["prob"], x["dist_to_target"]))
    return results


# ============================================================================
# HTTP HELPER
# ============================================================================
def http_get(url: str, headers: Dict = None) -> Optional[Dict]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        log.error("HTTP %d em %s: %s" % (e.code, url, e.read().decode()[:200]))
        return None
    except Exception as e:
        log.error("Erro em %s: %s" % (url, e))
        return None


# ============================================================================
# FOOTBALL-DATA.ORG — partidas e times
# ============================================================================
LEAGUE_NAMES = {
    "PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
    "BL1": "Bundesliga", "FL1": "Ligue 1", "DED": "Eredivisie",
    "PPL": "Liga Portugal", "ELC": "Championship", "SPL": "Scottish Premiership",
    "TUR": "Super Lig Turquia",
    "BEL": "Belgica - First Div A", "SUI": "Suica - Super League",
    "AUT": "Austria - Bundesliga", "GRE": "Grecia - Super League",
    "POL": "Polonia - Ekstraklasa", "DEN": "Dinamarca - Superliga",
    "NOR": "Noruega - Eliteserien", "SWE": "Suecia - Allsvenskan",
    "CL": "Champions League", "EL": "Europa League", "ECL": "Conference League",
    "EC": "Eurocopa", "NL": "Nations League", "WC": "Copa do Mundo",
    "BSA": "Brasileirao Serie A", "BSB": "Brasileirao Serie B", "CPB": "Copa do Brasil",
    "CLI": "Libertadores", "CSA": "Sul-Americana",
    "MLS": "MLS", "APD": "Argentina Primera", "MXN": "Liga MX", "COL": "Liga Colombia",
    "CHI": "Primera Chile", "URU": "Primera Uruguay",
    "JPL": "J-League", "KOR": "K League 1", "CSL": "Chinese Super League",
    "AUS": "A-League Australia",
}

LEAGUE_AVGS = {
    # liga: (media_gols_marcados, media_gols_sofridos)
    "PL": (1.45, 1.10), "PD": (1.35, 1.00), "SA": (1.30, 1.00),
    "BL1": (1.55, 1.15), "FL1": (1.25, 0.95), "DED": (1.55, 1.20),
    "PPL": (1.35, 1.05), "ELC": (1.40, 1.10), "SPL": (1.50, 1.15), "TUR": (1.45, 1.15),
    "BEL": (1.50, 1.15), "SUI": (1.50, 1.15), "AUT": (1.55, 1.20),
    "GRE": (1.30, 1.00), "POL": (1.30, 1.05), "DEN": (1.45, 1.15),
    "NOR": (1.45, 1.15), "SWE": (1.40, 1.10),
    "CL": (1.50, 1.10), "EL": (1.45, 1.10), "ECL": (1.40, 1.05),
    "EC": (1.25, 0.95), "NL": (1.30, 1.00), "WC": (1.20, 0.90),
    "BSA": (1.40, 1.05), "BSB": (1.35, 1.05), "CPB": (1.30, 1.00),
    "CLI": (1.35, 1.00), "CSA": (1.30, 1.00), "MLS": (1.50, 1.15),
    "APD": (1.45, 1.10), "MXN": (1.40, 1.05), "COL": (1.35, 1.05),
    "CHI": (1.30, 1.00), "URU": (1.35, 1.05), "JPL": (1.35, 1.00),
    "KOR": (1.35, 1.05), "CSL": (1.40, 1.05), "AUS": (1.45, 1.15),
}

SPORT_KEYS = {
    "PL": "soccer_epl", "PD": "soccer_spain_la_liga", "SA": "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga", "FL1": "soccer_france_ligue_one",
    "DED": "soccer_netherlands_eredivisie", "PPL": "soccer_portugal_primeira_liga",
    "ELC": "soccer_england_league1", "SPL": "soccer_scotland_premiership",
    "TUR": "soccer_turkey_super_league",
    "BEL": "soccer_belgium_first_div", "SUI": "soccer_switzerland_superleague",
    "AUT": "soccer_austria_bundesliga", "GRE": "soccer_greece_super_league",
    "POL": "soccer_poland_ekstraklasa", "DEN": "soccer_denmark_superliga",
    "NOR": "soccer_norway_eliteserien", "SWE": "soccer_sweden_allsvenskan",
    "CL": "soccer_uefa_champs_league", "EL": "soccer_uefa_europa_league",
    "ECL": "soccer_uefa_europa_conference_league", "EC": "soccer_uefa_european_championship",
    "NL": "soccer_uefa_nations_league", "WC": "soccer_fifa_world_cup",
    "BSA": "soccer_brazil_campeonato", "BSB": "soccer_brazil_serie_b",
    "CPB": "soccer_brazil_copa_do_brasil", "CLI": "soccer_conmebol_copa_libertadores",
    "CSA": "soccer_conmebol_copa_sudamericana", "MLS": "soccer_usa_mls",
    "APD": "soccer_argentina_primera_division", "MXN": "soccer_mexico_ligamx",
    "COL": "soccer_colombia_primera_a", "CHI": "soccer_chile_campeonato",
    "JPL": "soccer_japan_j_league", "KOR": "soccer_korea_kleague1",
    "CSL": "soccer_china_superleague", "AUS": "soccer_australia_aleague",
}


def _iso_to_brt(iso_str: str) -> str:
    try:
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=3)
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return iso_str[:16] if iso_str else "?"


def fetch_matches(api_key: str, league_ids: List[str], days: int = 2) -> List[Dict]:
    """Busca partidas dos proximos N dias em todas as ligas configuradas."""
    today = date.today()
    end   = today + timedelta(days=days)
    headers = {"X-Auth-Token": api_key}
    matches = []

    for lid in league_ids:
        lid = lid.strip()
        if not lid:
            continue
        url = (
            "https://api.football-data.org/v4/competitions/%s/matches"
            "?dateFrom=%s&dateTo=%s&status=SCHEDULED"
        ) % (lid, today.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        data = http_get(url, headers)
        if not data:
            continue
        found = data.get("matches", [])
        for m in found:
            home = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "?")
            away = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "?")
            utc  = m.get("utcDate", "")
            try:
                dt = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=3)
                date_str = dt.strftime("%d/%m %H:%M")
            except Exception:
                date_str = utc[:16]
            matches.append({
                "homeTeam": home, "awayTeam": away,
                "league": LEAGUE_NAMES.get(lid, lid), "league_id": lid,
                "date": date_str,
                "home_id": m.get("homeTeam", {}).get("id"),
                "away_id": m.get("awayTeam", {}).get("id"),
            })
        log.info("  %s (%s): %d partidas" % (lid, LEAGUE_NAMES.get(lid, lid), len(found)))

    log.info("Total de partidas encontradas: %d" % len(matches))
    return matches


def fetch_team_form(api_key: str, team_id: int, n: int = 6) -> Dict:
    """Historico real do time com peso exponencial nos jogos recentes."""
    url = "https://api.football-data.org/v4/teams/%d/matches?limit=%d&status=FINISHED" % (team_id, n)
    data = http_get(url, {"X-Auth-Token": api_key})
    empty = {"scored": 1.3, "conceded": 1.3, "form_score": 0.5,
             "home_scored": 1.4, "home_conceded": 1.2,
             "away_scored": 1.1, "away_conceded": 1.4, "sample": 0, "form_str": "-----"}
    if not data or not data.get("matches"):
        return empty

    matches = data["matches"]
    scored_l, conceded_l = [], []
    home_s, home_c, away_s, away_c = [], [], [], []
    form_pts = []

    for m in matches:
        score = m.get("score", {}).get("fullTime", {})
        hs = score.get("home", 0) or 0
        as_ = score.get("away", 0) or 0
        is_home = m.get("homeTeam", {}).get("id") == team_id
        s, c = (hs, as_) if is_home else (as_, hs)
        scored_l.append(s); conceded_l.append(c)
        (home_s if is_home else away_s).append(s)
        (home_c if is_home else away_c).append(c)
        form_pts.append(3 if s > c else (1 if s == c else 0))

    ng = len(scored_l)
    if ng == 0:
        return empty

    ws = [0.5 ** (ng - 1 - i) for i in range(ng)]

    def wavg(lst, w):
        if not lst: return 0.0
        w2 = w[:len(lst)]; tw = sum(w2)
        return sum(v * ww for v, ww in zip(lst, w2)) / tw if tw > 0 else sum(lst) / len(lst)

    return {
        "scored": round(wavg(scored_l, ws), 3),
        "conceded": round(wavg(conceded_l, ws), 3),
        "form_score": round(wavg(form_pts, ws) / 3.0, 3),
        "home_scored": round(sum(home_s) / max(1, len(home_s)), 3),
        "home_conceded": round(sum(home_c) / max(1, len(home_c)), 3),
        "away_scored": round(sum(away_s) / max(1, len(away_s)), 3),
        "away_conceded": round(sum(away_c) / max(1, len(away_c)), 3),
        "sample": ng,
        "form_str": "".join("V" if p == 3 else "E" if p == 1 else "D" for p in form_pts[-5:]),
    }


# ============================================================================
# THE ODDS API — odds reais incluindo handicap asiatico
# ============================================================================
def fetch_odds(api_key: str, sport_key: str) -> List[Dict]:
    """
    Busca odds via The Odds API e calcula CONSENSO (mediana) entre todas
    as casas retornadas. Mantem exatamente 2 chamadas por liga (mesma cota
    de antes) mas agora captura:
      - Varias linhas de Over/Under do jogo todo (0.5/1.5/2.5/3.5)
      - Over/Under do PRIMEIRO TEMPO (0.5/1.5), quando a casa oferece
    Cada linha retornada pelas casas e arredondada para a linha-alvo
    mais proxima — nao usamos "alternate_totals" (custaria uma chamada
    extra por liga e estouraria a cota gratuita de 500/mes).
    """
    import statistics

    FT_LINES = [0.5, 1.5, 2.5, 3.5]
    H1_LINES = [0.5, 1.5]

    def nearest_line(pt, lines):
        return min(lines, key=lambda l: abs(l - pt))

    url = (
        "https://api.the-odds-api.com/v4/sports/%s/odds"
        "?apiKey=%s&regions=eu&markets=h2h,totals,btts,totals_h1"
        "&oddsFormat=decimal&dateFormat=iso"
    ) % (sport_key, api_key)
    data = http_get(url)
    if not data or not isinstance(data, list):
        return []

    url_ah = (
        "https://api.the-odds-api.com/v4/sports/%s/odds"
        "?apiKey=%s&regions=eu&markets=spreads"
        "&oddsFormat=decimal&dateFormat=iso"
    ) % (sport_key, api_key)
    data_ah = http_get(url_ah) or []
    ah_index = {}
    if isinstance(data_ah, list):
        for g in data_ah:
            ah_index[(g.get("home_team", ""), g.get("away_team", ""))] = g

    def median(lst):
        return round(statistics.median(lst), 3) if lst else None

    result = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        buckets = {"home": [], "draw": [], "away": [], "btts": []}
        for line in FT_LINES:
            key = str(line).replace(".", "")
            buckets["over%s" % key]  = []
            buckets["under%s" % key] = []
        for line in H1_LINES:
            key = str(line).replace(".", "")
            buckets["h1_over%s" % key]  = []
            buckets["h1_under%s" % key] = []

        for bm in game.get("bookmakers", [])[:8]:
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == home:
                            buckets["home"].append(o["price"])
                        elif o["name"] == away:
                            buckets["away"].append(o["price"])
                        elif o["name"] == "Draw":
                            buckets["draw"].append(o["price"])

                elif mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        pt = o.get("point")
                        if pt is None:
                            continue
                        line = nearest_line(pt, FT_LINES)
                        lkey = str(line).replace(".", "")
                        if o["name"] == "Over":
                            buckets["over%s" % lkey].append(o["price"])
                        elif o["name"] == "Under":
                            buckets["under%s" % lkey].append(o["price"])

                elif mkt["key"] == "totals_h1":
                    for o in mkt.get("outcomes", []):
                        pt = o.get("point")
                        if pt is None:
                            continue
                        line = nearest_line(pt, H1_LINES)
                        lkey = str(line).replace(".", "")
                        if o["name"] == "Over":
                            buckets["h1_over%s" % lkey].append(o["price"])
                        elif o["name"] == "Under":
                            buckets["h1_under%s" % lkey].append(o["price"])

                elif mkt["key"] == "btts":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == "Yes":
                            buckets["btts"].append(o["price"])

        odds = {k: median(v) for k, v in buckets.items()}
        odds.update({"ah_home_m5": None, "ah_home_p5": None,
                      "ah_away_m5": None, "ah_away_p5": None})

        ah_buckets = {"ah_home_m5": [], "ah_home_p5": [], "ah_away_m5": [], "ah_away_p5": []}
        ah_game = ah_index.get((home, away))
        if ah_game:
            for bm in ah_game.get("bookmakers", [])[:6]:
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "spreads":
                        continue
                    for o in mkt.get("outcomes", []):
                        pt = o.get("point", 999)
                        name = o.get("name", "")
                        if abs(pt - (-0.5)) < 0.01 and name == home:
                            ah_buckets["ah_home_m5"].append(o["price"])
                        elif abs(pt - 0.5) < 0.01 and name == home:
                            ah_buckets["ah_home_p5"].append(o["price"])
                        elif abs(pt - (-0.5)) < 0.01 and name == away:
                            ah_buckets["ah_away_m5"].append(o["price"])
                        elif abs(pt - 0.5) < 0.01 and name == away:
                            ah_buckets["ah_away_p5"].append(o["price"])
        for k, v in ah_buckets.items():
            odds[k] = median(v)

        result.append({
            "home": home, "away": away, "odds": odds,
            "n_bookmakers": len(game.get("bookmakers", [])),
            "commence_time": game.get("commence_time", ""),
        })

    log.info("  %s: %d jogos com odds (consenso, multi-linha)" % (sport_key, len(result)))
    return result


def normalize_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s)


def names_match(name_a: str, name_b: str) -> int:
    """Score de 0-5 indicando o quanto dois nomes de time se parecem."""
    na, nb = normalize_name(name_a), normalize_name(name_b)
    score = 0
    if na in nb or nb in na: score += 3
    if na[:4] == nb[:4]: score += 1
    if na[:6] == nb[:6]: score += 1
    return score


def match_odds(home_name: str, away_name: str, odds_list: List[Dict]) -> Optional[Dict]:
    nh, na = normalize_name(home_name), normalize_name(away_name)
    best, best_score = None, 0
    for o in odds_list:
        oh, oa = normalize_name(o["home"]), normalize_name(o["away"])
        score = 0
        if nh in oh or oh in nh: score += 2
        if na in oa or oa in na: score += 2
        if nh[:4] in oh: score += 1
        if na[:4] in oa: score += 1
        if score > best_score:
            best_score, best = score, o
    return best["odds"] if best and best_score >= 3 else None


# ============================================================================
# GEMINI — analise qualitativa
# ============================================================================
def _gemini_sync(api_key: str, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text or ""


async def call_gemini(api_key: str, prompt: str) -> str:
    if not api_key:
        return ""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _gemini_sync, api_key, prompt)
        log.info("Gemini OK: %d chars" % len(result))
        return result
    except Exception as e:
        log.error("Gemini ERRO: %s" % e)
        return ""


# ============================================================================
# TELEGRAM
# ============================================================================
def tg_send_sync(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        log.error("Telegram nao configurado")
        return False
    url = "https://api.telegram.org/bot%s/sendMessage" % token
    payload = json.dumps({
        "chat_id": chat_id, "text": text[:4096],
        "parse_mode": "HTML", "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode()).get("ok", False)
    except urllib.error.HTTPError as e:
        log.error("Telegram %d: %s" % (e.code, e.read().decode()[:200]))
        return False
    except Exception as e:
        log.error("Telegram erro: %s" % e)
        return False


async def tg_send(token, chat_id, text) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, tg_send_sync, token, chat_id, text)


def tg_get_updates_sync(token, offset=0):
    url = "https://api.telegram.org/bot%s/getUpdates?offset=%d&timeout=5" % (token, offset)
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("result", []) if data.get("ok") else []
    except Exception:
        return []


async def send_entry(token, chat_id, entry, cfg: Config) -> bool:
    nav = BETANO_PATH.get(entry["market_key"], ("Principais", entry["market_label"], entry["market_label"]))
    tab, section, option = nav
    retorno = entry["stake"] * entry["odds"]
    lucro   = entry["stake"] * (entry["odds"] - 1)

    msg = (
        "FAVORITO DE ALTA PROBABILIDADE\n"
        "------------------------------\n\n"
        "<b>%s vs %s</b>\n"
        "%s | %s\n\n"
        "MERCADO: %s\n"
        "ODD: %.2f\n"
        "PROBABILIDADE REAL: %.1f%%\n"
        "PROBABILIDADE IMPLICITA: %.1f%%\n\n"
        "APOSTAR: R$ %.2f\n"
        "RETORNO: R$ %.2f | LUCRO: R$ %.2f\n\n"
        "BETANO:\n"
        "<code>Futebol -> %s\n"
        "-> %s vs %s\n"
        "-> %s -> %s\n"
        "-> %s</code>\n\n"
        "%s"
    ) % (
        entry["home_team"], entry["away_team"],
        entry["league"], entry["date"],
        entry["market_label"], entry["odds"],
        entry["prob"] * 100, entry["implied"] * 100,
        entry["stake"], retorno, lucro,
        entry["league"], entry["home_team"], entry["away_team"],
        tab, section, option,
        datetime.now().strftime("%H:%M:%S"),
    )
    ok = await tg_send(token, chat_id, msg)

    if ok and cfg.gemini_api_key:
        await asyncio.sleep(1)
        prompt = (
            "Analista de apostas esportivas. Tecnico e direto.\n"
            "%s vs %s - %s\n"
            "Mercado escolhido: %s @ %.2f\n"
            "Probabilidade calculada pelo modelo: %.1f%%\n\n"
            "Responda em 2 partes curtas (max 100 palavras):\n"
            "1. POR QUE E FAVORITO: o que sustenta essa alta probabilidade\n"
            "2. RISCO: o unico fator que poderia surpreender"
        ) % (entry["home_team"], entry["away_team"], entry["league"],
             entry["market_label"], entry["odds"], entry["prob"] * 100)
        ai_txt = await call_gemini(cfg.gemini_api_key, prompt)
        if ai_txt:
            await tg_send(token, chat_id, "ANALISE DA IA\n-------------\n\n%s" % ai_txt[:2000])

    return ok


async def send_diagnostic(token, chat_id, diag: Dict, cfg: Config) -> bool:
    matches_txt = "\n".join("  - %s" % m for m in diag["matches_list"][:15]) or "  Nenhuma partida"
    near_txt = "\n".join("  %s" % m for m in diag["near_misses"][:10]) or "  Nenhum quase-favorito"
    msg = (
        "DIAGNOSTICO DO SCAN\n"
        "-------------------\n"
        "%s\n\n"
        "Partidas verificadas: %d\n"
        "Com odds: %d\n"
        "Faixa alvo: %.2f - %.2f | Prob minima: %.0f%%\n"
        "Entradas encontradas: %d\n\n"
        "Jogos verificados:\n%s\n\n"
        "Quase-favoritos (fora da faixa ou prob baixa):\n%s"
    ) % (
        datetime.now().strftime("%d/%m %H:%M"),
        diag["matches_found"], diag["with_odds"],
        cfg.odd_min, cfg.odd_max, cfg.min_prob * 100,
        diag["entries_found"],
        matches_txt, near_txt,
    )
    return await tg_send(token, chat_id, msg)


async def send_startup(token, chat_id, cfg: Config) -> bool:
    # Estimativa de uso da The Odds API (free tier = 500 req/mes)
    # 2 chamadas (mercados + handicap) por liga ativa em cada scan
    scans_por_dia = (24 * 60) / max(cfg.scan_interval_min, 1)
    estimativa_mes = int(len(cfg.league_ids) * 2 * scans_por_dia * 30)
    alerta_cota = (
        "\nATENCAO: estimativa de %d chamadas/mes na Odds API (limite gratis: 500).\n"
        "Aumente SCAN_INTERVAL_MIN ou reduza LEAGUE_IDS para nao estourar a cota."
        % estimativa_mes
    ) if estimativa_mes > 500 else "\nUso estimado na Odds API: %d chamadas/mes (dentro do limite gratis de 500)." % estimativa_mes

    msg = (
        "BETTING AI ENGINE - ONLINE\n"
        "--------------------------\n\n"
        "Estrategia: Favoritos de alta probabilidade\n"
        "Mercados: 1X2, Over/Under (0.5 a 3.5), 1o Tempo, Ambas Marcam, Handicap Asiatico\n"
        "Faixa de odd alvo: %.2f - %.2f\n"
        "Probabilidade minima: %.0f%%\n"
        "Banca: R$ %.2f\n"
        "Max entradas por scan: %d\n"
        "Scan a cada %d min\n"
        "Ligas ativas: %d\n"
        "%s\n\n"
        "%s"
    ) % (
        cfg.odd_min, cfg.odd_max, cfg.min_prob * 100,
        cfg.banca, cfg.max_entries, cfg.scan_interval_min,
        len(cfg.league_ids), alerta_cota,
        datetime.now().strftime("%d/%m/%Y %H:%M"),
    )
    return await tg_send(token, chat_id, msg)


async def send_error(token, chat_id, error) -> bool:
    return await tg_send(token, chat_id,
        "ERRO\n<code>%s</code>\n%s" % (str(error)[:400], datetime.now().strftime("%d/%m %H:%M")))


async def send_auto_results(token, chat_id, resolved: list) -> bool:
    """Notifica resultados resolvidos automaticamente apos o jogo terminar."""
    if not resolved:
        return False
    lines = []
    for r in resolved:
        tag = "GANHOU" if r["outcome"] == "win" else "PERDEU"
        lines.append(
            "%s - %s\n  %s | placar %s\n  stake R$ %.2f -> retorno R$ %.2f"
            % (tag, r["match"], r["market"], r["score"], r["stake"], r["returned"])
        )
    banca_final = resolved[-1]["banca"]
    msg = (
        "RESULTADOS AUTOMATICOS\n"
        "-----------------------\n\n"
        + "\n\n".join(lines) +
        "\n\nBanca atualizada: R$ %.2f" % banca_final
    )
    return await tg_send(token, chat_id, msg)


async def send_backtest(token, chat_id, memory, days: int = 30) -> bool:
    bt = memory.get_backtest(days)
    if not bt:
        return await tg_send(token, chat_id,
            "BACKTEST (%d dias)\n-------------------\nSem resultados resolvidos ainda nesse periodo." % days)

    o = bt["overall"]
    by_league_txt = "\n".join(
        "  %s: %dV %dD | ROI %+.1f%%" % (lg, s["wins"], s["losses"], s["roi"])
        for lg, s in sorted(bt["by_league"].items(), key=lambda x: -(x[1]["wins"] + x[1]["losses"]))[:8]
    ) or "  sem dados"
    by_market_txt = "\n".join(
        "  %s: %dV %dD | ROI %+.1f%%" % (mk, s["wins"], s["losses"], s["roi"])
        for mk, s in sorted(bt["by_market"].items(), key=lambda x: -(x[1]["wins"] + x[1]["losses"]))[:8]
    ) or "  sem dados"

    msg = (
        "BACKTEST - ULTIMOS %d DIAS\n"
        "---------------------------\n\n"
        "Total resolvidas: %d\n"
        "Vitorias: %d | Derrotas: %d\n"
        "Win rate: %.1f%%\n"
        "Apostado: R$ %.2f | Retornado: R$ %.2f\n"
        "ROI: %+.1f%%\n\n"
        "POR LIGA:\n%s\n\n"
        "POR MERCADO:\n%s"
    ) % (
        bt["days"], bt["total_rows"], o["wins"], o["losses"], o["win_rate"],
        o["staked"], o["returned"], o["roi"], by_league_txt, by_market_txt,
    )
    return await tg_send(token, chat_id, msg)


# ============================================================================
# SCANNER PRINCIPAL
# ============================================================================
async def scan(cfg: Config, memory) -> Tuple[List[Dict], Dict]:
    log.info("Scan iniciado - %d ligas configuradas" % len(cfg.league_ids))
    diag = {"matches_found": 0, "with_odds": 0, "matches_list": [],
            "near_misses": [], "entries_found": 0, "leagues_fallback": []}

    loop = asyncio.get_event_loop()
    matches_raw = await loop.run_in_executor(_executor, fetch_matches, cfg.football_data_key, cfg.league_ids, 2)

    # Buscar odds de TODAS as ligas configuradas (sempre, mesmo sem fixture do football-data)
    odds_by_league = {}
    for lid in cfg.league_ids:
        lid = lid.strip()
        sport_key = SPORT_KEYS.get(lid)
        if not sport_key:
            continue
        odds_list = await loop.run_in_executor(_executor, fetch_odds, cfg.odds_api_key, sport_key)
        odds_by_league[lid] = odds_list

    # football-data.org free so cobre ~12 ligas. Para as demais (Libertadores,
    # MLS, Liga MX, Argentina, J-League etc.) usamos os proprios jogos
    # retornados pela The Odds API como fonte de partidas (sem form/Elo,
    # cai automaticamente para media da liga).
    leagues_with_fd = set(m.get("league_id", "") for m in matches_raw)
    all_matches = list(matches_raw)

    for lid in cfg.league_ids:
        lid = lid.strip()
        if not lid or lid in leagues_with_fd:
            continue
        odds_list = odds_by_league.get(lid, [])
        if not odds_list:
            continue
        diag["leagues_fallback"].append(lid)
        for g in odds_list:
            if not g.get("home") or not g.get("away"):
                continue
            all_matches.append({
                "homeTeam": g["home"], "awayTeam": g["away"],
                "league": LEAGUE_NAMES.get(lid, lid), "league_id": lid,
                "date": _iso_to_brt(g.get("commence_time", "")),
                "home_id": None, "away_id": None,
            })

    diag["matches_found"] = len(all_matches)
    if not all_matches:
        return [], diag

    entries = []
    for m in all_matches:
        name = "%s vs %s" % (m["homeTeam"], m["awayTeam"])
        diag["matches_list"].append(name)
        lid = m.get("league_id", "")

        odds_list = odds_by_league.get(lid, [])
        odds = match_odds(m["homeTeam"], m["awayTeam"], odds_list)
        if not odds or not any(v for v in odds.values() if v):
            continue
        diag["with_odds"] += 1

        avg_s, avg_c = LEAGUE_AVGS.get(lid, (1.35, 1.05))
        home_id, away_id = m.get("home_id"), m.get("away_id")

        home_form = away_form = None
        if cfg.football_data_key and home_id and away_id:
            try:
                home_form = await loop.run_in_executor(_executor, fetch_team_form, cfg.football_data_key, home_id, 6)
                away_form = await loop.run_in_executor(_executor, fetch_team_form, cfg.football_data_key, away_id, 6)
            except Exception as e:
                log.warning("Erro form: %s" % e)

        if home_form and home_form["sample"] >= 3 and away_form and away_form["sample"] >= 3:
            home_attack  = (home_form["home_scored"] or home_form["scored"]) * (0.85 + 0.30 * home_form["form_score"])
            home_defense = (home_form["home_conceded"] or home_form["conceded"]) * (1.15 - 0.30 * home_form["form_score"])
            away_attack  = (away_form["away_scored"] or away_form["scored"]) * (0.85 + 0.30 * away_form["form_score"])
            away_defense = (away_form["away_conceded"] or away_form["conceded"]) * (1.15 - 0.30 * away_form["form_score"])
            data_source = "real"
        else:
            home_attack, home_defense = avg_s * 1.10, avg_c * 0.90
            away_attack, away_defense = avg_s * 0.90, avg_c * 1.10
            data_source = "liga"

        # Ajuste por Elo Rating — time mais forte ataca mais e defende melhor
        elo_home = memory.elo_strength_factor(home_id, m["homeTeam"]) if home_id else 1.0
        elo_away = memory.elo_strength_factor(away_id, m["awayTeam"]) if away_id else 1.0
        home_attack  *= elo_home
        home_defense /= elo_home
        away_attack  *= elo_away
        away_defense /= elo_away

        favorites = find_favorite_markets(home_attack, home_defense, away_attack, away_defense, odds, cfg)

        if favorites:
            best = favorites[0]
            entries.append({
                "home_team": m["homeTeam"], "away_team": m["awayTeam"],
                "league": m["league"], "league_id": lid, "date": m["date"],
                "data_source": data_source,
                "market_key": best["key"], "market_label": best["label"],
                "odds": best["odds"], "prob": best["prob"], "implied": best["implied"],
                "ev_pct": best["ev_pct"], "edge_pct": best["edge_pct"],
                "stake": best["stake"],
            })
            diag["entries_found"] += 1
            log.info("  FAVORITO: %s - %s @ %.2f (prob %.1f%%)" % (
                name, best["label"], best["odds"], best["prob"] * 100))
        else:
            # Registrar o mercado mais proximo do alvo para diagnostico
            lh = home_attack * away_defense
            la = away_attack * home_defense
            probs = match_probs(lh, la)
            closest = None
            closest_dist = 999
            for key, prob in probs.items():
                o = odds.get(key)
                if not o or o <= 1.01:
                    continue
                d = abs(o - 1.50)
                if d < closest_dist:
                    closest_dist, closest = d, (key, prob, o)
            if closest:
                k, p, o = closest
                diag["near_misses"].append(
                    "%s: %s @ %.2f (prob %.1f%%)" % (name, MARKET_LABELS.get(k, k), o, p * 100)
                )

    entries.sort(key=lambda x: -x["prob"])
    entries = entries[:cfg.max_entries]

    log.info("Scan finalizado: %d favoritos de %d partidas (%d com odds, %d ligas via fallback Odds API)" % (
        len(entries), diag["matches_found"], diag["with_odds"], len(diag["leagues_fallback"])))
    return entries, diag


# ============================================================================
# MEMORIA — banca e historico
# ============================================================================

# ============================================================================
# ELO RATING — forca relativa de cada time, atualizado a cada resultado
# ============================================================================
ELO_BASE = 1500.0
ELO_K = 24.0           # velocidade de ajuste por jogo
ELO_HOME_ADV = 60.0    # bonus de mando de campo em pontos Elo


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Probabilidade esperada de A vencer B dado os ratings (formula classica do Elo)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = ELO_K) -> float:
    """
    score_a: 1.0 = A venceu, 0.5 = empate, 0.0 = A perdeu
    Retorna o novo rating de A.
    """
    expected = elo_expected(rating_a, rating_b)
    return rating_a + k * (score_a - expected)


def elo_goal_multiplier(goal_diff: int) -> float:
    """Vitorias por margem maior pesam mais no ajuste do Elo (padrao usado no futebol)."""
    gd = abs(goal_diff)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return 1.5 + (gd - 2) / 8.0


# ============================================================================
# RESULTADOS FINALIZADOS — alimenta Elo e resolve apostas automaticamente
# ============================================================================
def fetch_finished_matches(api_key: str, league_ids: List[str], days_back: int = 3) -> List[Dict]:
    today = date.today()
    start = today - timedelta(days=days_back)
    headers = {"X-Auth-Token": api_key}
    finished = []
    for lid in league_ids:
        lid = lid.strip()
        if not lid:
            continue
        url = (
            "https://api.football-data.org/v4/competitions/%s/matches"
            "?dateFrom=%s&dateTo=%s&status=FINISHED"
        ) % (lid, start.isoformat(), today.isoformat())
        data = http_get(url, headers)
        if not data:
            continue
        for m in data.get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            hg, ag = score.get("home"), score.get("away")
            if hg is None or ag is None:
                continue
            finished.append({
                "league_id": lid,
                "home_id": m.get("homeTeam", {}).get("id"),
                "home_name": m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "?"),
                "away_id": m.get("awayTeam", {}).get("id"),
                "away_name": m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "?"),
                "home_goals": hg, "away_goals": ag,
            })
    return finished


def evaluate_market_outcome(market_key: str, hg: int, ag: int) -> Optional[bool]:
    """Retorna True (ganhou), False (perdeu) ou None (mercado nao reconhecido)."""
    if market_key == "home":        return hg > ag
    if market_key == "draw":        return hg == ag
    if market_key == "away":        return ag > hg
    if market_key == "over25":      return (hg + ag) > 2.5
    if market_key == "under25":     return (hg + ag) < 2.5
    if market_key == "btts":        return hg > 0 and ag > 0
    if market_key == "ah_home_m5":  return hg > ag
    if market_key == "ah_home_p5":  return hg >= ag
    if market_key == "ah_away_m5":  return ag > hg
    if market_key == "ah_away_p5":  return ag >= hg
    return None


# ============================================================================
# BANCO SQLITE — historico estruturado de previsoes, odds e resultados
# ============================================================================
import sqlite3


class Database:
    def __init__(self, path="data/betting.db"):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS tips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_key TEXT NOT NULL,
                home_team TEXT, away_team TEXT, league TEXT, league_id TEXT,
                market_key TEXT, market_label TEXT,
                odds_at_alert REAL, odds_now REAL,
                prob_real REAL, implied REAL, ev_pct REAL,
                stake REAL, date_sent TEXT,
                outcome TEXT, returned REAL DEFAULT 0,
                data_source TEXT
            )
        """)
        # Migracao leve: adiciona league_id se o banco for de uma versao anterior
        try:
            c.execute("ALTER TABLE tips ADD COLUMN league_id TEXT")
        except sqlite3.OperationalError:
            pass  # coluna ja existe
        c.execute("""
            CREATE TABLE IF NOT EXISTS elo (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT,
                rating REAL DEFAULT 1500.0,
                games INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS config_kv (
                k TEXT PRIMARY KEY, v TEXT
            )
        """)
        self.conn.commit()

        old_json = "data/history.json"
        c.execute("SELECT COUNT(*) AS n FROM tips")
        n_tips = c.fetchone()["n"]
        if n_tips == 0 and os.path.exists(old_json):
            try:
                old = json.load(open(old_json, "r", encoding="utf-8"))
                for t in old.get("tips", []):
                    c.execute(
                        "INSERT INTO tips (match_key, home_team, away_team, league, market_key, "
                        "odds_at_alert, stake, date_sent, outcome, returned) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (t.get("match",""), t.get("home_team",""), t.get("away_team",""),
                         t.get("league",""), t.get("market",""), t.get("odds",0),
                         t.get("stake",0), t.get("date",""), t.get("outcome"), t.get("returned",0) or 0)
                    )
                if old.get("banca") is not None:
                    self.set_kv("banca", str(old["banca"]))
                self.conn.commit()
                log.info("Migrado %d tips do JSON antigo para SQLite" % len(old.get("tips",[])))
            except Exception as e:
                log.warning("Falha ao migrar JSON antigo: %s" % e)

    def get_kv(self, key, default=None):
        c = self.conn.cursor()
        c.execute("SELECT v FROM config_kv WHERE k=?", (key,))
        row = c.fetchone()
        return row["v"] if row else default

    def set_kv(self, key, value):
        c = self.conn.cursor()
        c.execute("INSERT INTO config_kv (k,v) VALUES (?,?) "
                   "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, str(value)))
        self.conn.commit()

    def already_sent_today(self, match_key) -> bool:
        today = date.today().isoformat()
        c = self.conn.cursor()
        c.execute("SELECT 1 FROM tips WHERE match_key=? AND date_sent=?", (match_key, today))
        return c.fetchone() is not None

    def record(self, match_key, entry):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO tips (match_key, home_team, away_team, league, league_id, market_key, market_label, "
            "odds_at_alert, odds_now, prob_real, implied, ev_pct, stake, date_sent, outcome, data_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (match_key, entry.get("home_team",""), entry.get("away_team",""), entry.get("league",""),
             entry.get("league_id",""),
             entry.get("market_key",""), entry.get("market_label",""),
             entry.get("odds",0), entry.get("odds",0),
             entry.get("prob",0), entry.get("implied",0), entry.get("ev_pct",0),
             entry.get("stake",0), date.today().isoformat(), None, entry.get("data_source",""))
        )
        self.conn.commit()

    def update_odds_now(self, match_key, odds_now):
        c = self.conn.cursor()
        c.execute("UPDATE tips SET odds_now=? WHERE match_key=? AND outcome IS NULL", (odds_now, match_key))
        self.conn.commit()

    def get_pending(self, days_back=3):
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        c = self.conn.cursor()
        c.execute("SELECT * FROM tips WHERE outcome IS NULL AND date_sent>=?", (cutoff,))
        return [dict(r) for r in c.fetchall()]

    def set_outcome(self, tip_id, outcome, returned):
        c = self.conn.cursor()
        c.execute("UPDATE tips SET outcome=?, returned=? WHERE id=?", (outcome, returned, tip_id))
        self.conn.commit()

    def get_banca(self, cfg_banca):
        v = self.get_kv("banca")
        return float(v) if v is not None else cfg_banca

    def set_banca(self, valor):
        self.set_kv("banca", round(float(valor), 2))

    def registrar_resultado(self, ganhou, valor):
        banca = self.get_banca(0)
        nova = round(banca + valor, 2) if ganhou else round(max(0, banca - valor), 2)
        self.set_banca(nova)
        return nova

    def get_stats(self):
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) AS n FROM tips")
        sent = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) AS n, COALESCE(SUM(stake),0) AS staked FROM tips WHERE outcome='loss'")
        row = c.fetchone(); losses, staked = row["n"], row["staked"]
        c.execute("SELECT COUNT(*) AS n, COALESCE(SUM(returned),0) AS returned FROM tips WHERE outcome='win'")
        row = c.fetchone(); wins, returned = row["n"], row["returned"]
        total = wins + losses
        wr = (wins/total*100) if total > 0 else 0
        roi = ((returned-staked)/staked*100) if staked > 0 else 0
        return {
            "banca": self.get_banca(0), "enviados": sent,
            "wins": wins, "losses": losses, "total": total,
            "win_rate": round(wr,1), "roi": round(roi,1),
            "staked": round(staked,2), "returned": round(returned,2),
        }

    def get_backtest(self, days=30):
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        c = self.conn.cursor()
        c.execute(
            "SELECT league, market_label, odds_at_alert, outcome, stake, returned "
            "FROM tips WHERE date_sent>=? AND outcome IS NOT NULL", (cutoff,)
        )
        rows = [dict(r) for r in c.fetchall()]
        if not rows:
            return None

        def summarize(rows_subset):
            w = sum(1 for r in rows_subset if r["outcome"] == "win")
            l = sum(1 for r in rows_subset if r["outcome"] == "loss")
            staked = sum(r["stake"] for r in rows_subset if r["outcome"] in ("win","loss"))
            returned = sum(r["returned"] for r in rows_subset if r["outcome"] == "win")
            roi = ((returned-staked)/staked*100) if staked > 0 else 0
            wr = (w/(w+l)*100) if (w+l) > 0 else 0
            return {"wins": w, "losses": l, "win_rate": round(wr,1), "roi": round(roi,1),
                    "staked": round(staked,2), "returned": round(returned,2)}

        by_league = {}
        for r in rows: by_league.setdefault(r["league"], []).append(r)
        by_market = {}
        for r in rows: by_market.setdefault(r["market_label"], []).append(r)

        return {
            "overall": summarize(rows),
            "by_league": {k: summarize(v) for k, v in by_league.items()},
            "by_market": {k: summarize(v) for k, v in by_market.items()},
            "days": days, "total_rows": len(rows),
        }

    def get_elo(self, team_id, team_name=""):
        c = self.conn.cursor()
        c.execute("SELECT rating, games FROM elo WHERE team_id=?", (team_id,))
        row = c.fetchone()
        if row:
            return row["rating"], row["games"]
        c.execute("INSERT INTO elo (team_id, team_name, rating, games, updated_at) VALUES (?,?,?,?,?)",
                   (team_id, team_name, ELO_BASE, 0, datetime.now().isoformat()))
        self.conn.commit()
        return ELO_BASE, 0

    def update_elo_match(self, home_id, home_name, away_id, away_name, home_goals, away_goals):
        if home_id is None or away_id is None:
            return
        rh, gh = self.get_elo(home_id, home_name)
        ra, ga = self.get_elo(away_id, away_name)

        if home_goals > away_goals:   score_h = 1.0
        elif home_goals == away_goals: score_h = 0.5
        else:                          score_h = 0.0

        mult = elo_goal_multiplier(home_goals - away_goals)
        new_rh = elo_update(rh + ELO_HOME_ADV, ra, score_h, ELO_K * mult) - ELO_HOME_ADV
        new_ra = elo_update(ra, rh + ELO_HOME_ADV, 1.0 - score_h, ELO_K * mult)

        c = self.conn.cursor()
        c.execute("UPDATE elo SET rating=?, games=games+1, updated_at=? WHERE team_id=?",
                   (round(new_rh,1), datetime.now().isoformat(), home_id))
        c.execute("UPDATE elo SET rating=?, games=games+1, updated_at=? WHERE team_id=?",
                   (round(new_ra,1), datetime.now().isoformat(), away_id))
        self.conn.commit()

    def elo_strength_factor(self, team_id, team_name=""):
        rating, games = self.get_elo(team_id, team_name)
        if games < 5:
            return 1.0
        diff = rating - ELO_BASE
        factor = 1.0 + (diff / 400.0) * 0.30
        return max(0.70, min(1.35, factor))

    def process_finished_results(self, finished_matches: list) -> dict:
        """
        Para cada jogo finalizado:
          1. Atualiza o Elo dos dois times.
          2. Procura apostas pendentes (tips) que batem com esse jogo
             pelo nome dos times e resolve automaticamente (win/loss).
          3. Atualiza a banca com o resultado.
        Retorna um resumo do que foi processado.
        """
        resolved = []
        elo_updates = 0

        pending = self.get_pending(days_back=5)

        for fm in finished_matches:
            # 1. Elo
            self.update_elo_match(fm["home_id"], fm["home_name"],
                                   fm["away_id"], fm["away_name"],
                                   fm["home_goals"], fm["away_goals"])
            elo_updates += 1

            # 2. Resolver tips pendentes que correspondem a esse jogo
            for tip in pending:
                if tip.get("outcome") is not None:
                    continue
                sh = names_match(tip["home_team"], fm["home_name"])
                sa = names_match(tip["away_team"], fm["away_name"])
                if sh < 3 or sa < 3:
                    continue
                won = evaluate_market_outcome(tip["market_key"], fm["home_goals"], fm["away_goals"])
                if won is None:
                    continue
                returned = round(tip["stake"] * tip["odds_at_alert"], 2) if won else 0.0
                outcome = "win" if won else "loss"
                self.set_outcome(tip["id"], outcome, returned)

                # 3. Atualizar banca
                banca = self.get_banca(0)
                if won:
                    nova = round(banca + returned - tip["stake"], 2)
                else:
                    nova = round(max(0, banca - tip["stake"]), 2)
                self.set_banca(nova)

                resolved.append({
                    "match": "%s vs %s" % (tip["home_team"], tip["away_team"]),
                    "market": tip["market_label"], "outcome": outcome,
                    "score": "%d-%d" % (fm["home_goals"], fm["away_goals"]),
                    "stake": tip["stake"], "returned": returned, "banca": nova,
                })
                tip["outcome"] = outcome  # evita resolver de novo no mesmo loop

        return {"resolved": resolved, "elo_updates": elo_updates}


class Memory:
    """Wrapper fino sobre Database (SQLite) — mantem a mesma interface de antes."""
    def __init__(self, path="data/betting.db"):
        self.db = Database(path)

    def get_banca(self, cfg_banca): return self.db.get_banca(cfg_banca)
    def set_banca(self, valor): self.db.set_banca(valor)
    def registrar_resultado(self, ganhou, valor): return self.db.registrar_resultado(ganhou, valor)
    def already_sent_today(self, key): return self.db.already_sent_today(key)
    def record(self, key, entry): self.db.record(key, entry)
    def get_stats(self): return self.db.get_stats()
    def get_backtest(self, days=30): return self.db.get_backtest(days)
    def get_pending(self, days_back=3): return self.db.get_pending(days_back)
    def set_outcome(self, tip_id, outcome, returned): self.db.set_outcome(tip_id, outcome, returned)
    def update_odds_now(self, key, odds_now): self.db.update_odds_now(key, odds_now)
    def elo_strength_factor(self, team_id, team_name=""): return self.db.elo_strength_factor(team_id, team_name)
    def update_elo_match(self, *a, **kw): return self.db.update_elo_match(*a, **kw)
    def process_finished_results(self, finished_matches): return self.db.process_finished_results(finished_matches)


# ============================================================================
# COMANDOS TELEGRAM
# ============================================================================
async def process_commands(cfg: Config, memory: Memory):
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
        msg = upd.get("message", {})
        text = msg.get("text", "").strip()
        chat = str(msg.get("chat", {}).get("id", ""))

        if chat != cfg.telegram_chat_id:
            continue

        parts = text.split()
        cmd = parts[0].lower() if parts else ""

        if cmd == "/banca":
            if len(parts) < 2:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Uso: /banca 150.00")
                continue
            try:
                valor = float(parts[1].replace(",", "."))
                memory.set_banca(valor)
                cfg.banca = valor
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "Banca atualizada: R$ %.2f" % valor)
            except ValueError:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Valor invalido.")

        elif cmd == "/ganhou":
            if len(parts) < 2:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Uso: /ganhou 47.20 (lucro)")
                continue
            try:
                lucro = float(parts[1].replace(",", "."))
                nova = memory.registrar_resultado(True, lucro)
                cfg.banca = nova
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "Vitoria! +R$ %.2f | Banca: R$ %.2f" % (lucro, nova))
            except ValueError:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Valor invalido.")

        elif cmd == "/perdeu":
            if len(parts) < 2:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Uso: /perdeu 13.08 (valor apostado)")
                continue
            try:
                perda = float(parts[1].replace(",", "."))
                nova = memory.registrar_resultado(False, perda)
                cfg.banca = nova
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                    "Derrota. -R$ %.2f | Banca: R$ %.2f" % (perda, nova))
            except ValueError:
                await tg_send(cfg.telegram_token, cfg.telegram_chat_id, "Valor invalido.")

        elif cmd == "/status":
            s = memory.get_stats()
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                "STATUS\n------\n"
                "Banca: R$ %.2f\n"
                "Sinais enviados: %d\n"
                "Vitorias: %d | Derrotas: %d\n"
                "Win rate: %.1f%%\n"
                "ROI: %+.1f%%" % (
                    s["banca"], s["enviados"], s["wins"], s["losses"], s["win_rate"], s["roi"]))

        elif cmd == "/backtest":
            dias = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30
            await send_backtest(cfg.telegram_token, cfg.telegram_chat_id, memory, dias)

        elif cmd == "/ajuda":
            await tg_send(cfg.telegram_token, cfg.telegram_chat_id,
                "COMANDOS\n--------\n"
                "/banca 150.00 - atualiza banca\n"
                "/ganhou 47.20 - registra vitoria\n"
                "/perdeu 13.08 - registra derrota\n"
                "/status - ver banca e historico\n"
                "/backtest 30 - relatorio por liga e mercado\n"
                "/ajuda - esta mensagem\n\n"
                "Resultados tambem sao resolvidos automaticamente\n"
                "apos o fim de cada jogo, sem precisar digitar nada.")

    try:
        json.dump({"offset": offset}, open(offset_file, "w"))
    except Exception:
        pass


# ============================================================================
# LOOP PRINCIPAL
# ============================================================================
async def run_cycle(cfg: Config, memory: Memory):
    log.info("=" * 50)
    log.info("Ciclo - %s" % datetime.now().strftime("%d/%m/%Y %H:%M"))
    try:
        cfg.banca = memory.get_banca(cfg.banca)

        # 1. Processar jogos finalizados: atualiza Elo + resolve apostas pendentes
        try:
            finished = await asyncio.get_event_loop().run_in_executor(
                _executor, fetch_finished_matches, cfg.football_data_key, cfg.league_ids, 3
            )
            if finished:
                result = memory.process_finished_results(finished)
                if result["elo_updates"]:
                    log.info("Elo atualizado com %d jogos finalizados" % result["elo_updates"])
                if result["resolved"]:
                    cfg.banca = memory.get_banca(cfg.banca)
                    await send_auto_results(cfg.telegram_token, cfg.telegram_chat_id, result["resolved"])
        except Exception as e:
            log.warning("Erro ao processar resultados finalizados: %s" % e)

        # 2. Scan normal de favoritos
        entries, diag = await scan(cfg, memory)

        if cfg.debug_mode:
            await send_diagnostic(cfg.telegram_token, cfg.telegram_chat_id, diag, cfg)

        if not entries:
            log.info("Nenhum favorito encontrado nesse scan")
            return

        sent = 0
        for entry in entries:
            key = "%s-%s-%s" % (entry["home_team"], entry["away_team"], entry["market_key"])
            if memory.already_sent_today(key):
                continue
            ok = await send_entry(cfg.telegram_token, cfg.telegram_chat_id, entry, cfg)
            if ok:
                memory.record(key, entry)
                sent += 1
                await asyncio.sleep(2)
        log.info("%d entradas enviadas" % sent)

    except Exception as e:
        log.error("Erro no ciclo: %s" % e, exc_info=True)
        try:
            await send_error(cfg.telegram_token, cfg.telegram_chat_id, str(e))
        except Exception:
            pass


async def main():
    log.info("=" * 60)
    log.info("Iniciando Betting AI Engine - Favoritos")
    log.info("=" * 60)

    cfg = Config()
    try:
        cfg.validate()
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    memory = Memory()
    cfg.banca = memory.get_banca(cfg.banca)
    log.info("Banca: R$ %.2f | Odd alvo: %.2f-%.2f | Prob min: %.0f%%" % (
        cfg.banca, cfg.odd_min, cfg.odd_max, cfg.min_prob * 100))

    ok = await send_startup(cfg.telegram_token, cfg.telegram_chat_id, cfg)
    log.info("Telegram OK" if ok else "Telegram falhou")

    cycle = 0
    while True:
        cycle += 1
        log.info("--- Ciclo #%d ---" % cycle)
        await process_commands(cfg, memory)
        await run_cycle(cfg, memory)
        log.info("Aguardando %d minutos..." % cfg.scan_interval_min)
        for _ in range(cfg.scan_interval_min):
            await asyncio.sleep(60)
            await process_commands(cfg, memory)


if __name__ == "__main__":
    asyncio.run(main())
