"""
Notifier — Envia alertas formatados pelo Telegram
===================================================
Usa a API oficial do Telegram (Bot API) — gratuita e sem restrições.
"""

import logging
import aiohttp
from datetime import datetime
from typing import Dict
from config import Config

log = logging.getLogger(__name__)

BETANO_URL = "https://www.betano.com.br/sport/futebol/"

BETANO_PATH = {
    "home":    ("Principais", "Resultado Final",      "1 — Vitória Casa"),
    "draw":    ("Principais", "Resultado Final",      "X — Empate"),
    "away":    ("Principais", "Resultado Final",      "2 — Vitória Visitante"),
    "over25":  ("Gols",       "Total de Gols",        "Mais de 2.5"),
    "under25": ("Gols",       "Total de Gols",        "Menos de 2.5"),
    "btts":    ("Gols",       "Ambas Equipes Marcam", "Sim"),
}


class Notifier:
    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.base    = f"https://api.telegram.org/bot{token}"

    async def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Envia mensagem pelo Telegram. Retorna True se OK."""
        if not self.token or not self.chat_id:
            log.error("Telegram não configurado (token ou chat_id ausente)")
            return False

        url     = f"{self.base}/sendMessage"
        payload = {
            "chat_id":                  self.chat_id,
            "text":                     text[:4096],   # limite Telegram
            "parse_mode":               parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as r:
                    body = await r.json()
                    if r.status == 200 and body.get("ok"):
                        return True
                    else:
                        log.error(f"Telegram erro {r.status}: {body}")
                        # Token errado → logar mas não crashar
                        if r.status == 401:
                            log.error("TELEGRAM_TOKEN inválido! Verifique as variáveis de ambiente.")
                        elif r.status == 400 and "chat not found" in str(body):
                            log.error("TELEGRAM_CHAT_ID inválido! Verifique as variáveis de ambiente.")
                        return False
        except aiohttp.ClientError as e:
            log.error(f"Erro de rede ao enviar Telegram: {e}")
            return False
        except Exception as e:
            log.error(f"Erro inesperado no Telegram: {e}", exc_info=True)
            return False

    async def send_opportunity(self, opp: Dict, banca: float) -> bool:
        nav = BETANO_PATH.get(opp["market_key"],
              ("Principais", opp["market_label"], opp["market_label"]))
        tab, section, option = nav

        retorno  = opp["stake"] * opp["odds"]
        lucro    = opp["stake"] * (opp["odds"] - 1)
        odd_min  = round(opp["odds"] * 0.97, 2)

        alt = [m for m in opp.get("all_markets", [])
               if m["ev_pct"] > 0 and m["label"] != opp["market_label"]][:2]
        alt_text = "\n".join(
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
            f"📍 <b>COMO APOSTAR NA BETANO:</b>\n"
            f"<code>Futebol → {opp['league']}\n"
            f"→ {opp['home_team']} vs {opp['away_team']}\n"
            f"→ Aba: {tab} → {section}\n"
            f"→ Selecionar: {option}</code>\n\n"
            f"⚠️ Odd mínima: <b>{odd_min}</b>  |  ⏱ Até 5 min\n\n"
            f"<b>Outros mercados com valor:</b>\n{alt_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧮 Prob real: {opp['real_prob']*100:.1f}%  "
            f"Impl: {opp['implied_prob']*100:.1f}%\n"
            f"📊 Kelly: {opp['kelly_frac']*100:.2f}% da banca\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )
        return await self._send(msg)

    async def send_ai_analysis(self, opp: Dict) -> bool:
        if not opp.get("ai_analysis"):
            return False
        msg = (
            f"🤖 <b>ANÁLISE DA IA</b>\n"
            f"{opp['home_team']} vs {opp['away_team']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{opp['ai_analysis'][:3000]}"
        )
        return await self._send(msg)

    async def send_startup_message(self, cfg: Config) -> bool:
        stop_val = cfg.banca * cfg.stop_loss_pct / 100
        msg = (
            f"🟢 <b>BETTING AI ENGINE — ONLINE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⚙️ <b>Configuração ativa:</b>\n"
            f"💰 Banca: R$ {cfg.banca:.2f}\n"
            f"🛑 Stop loss: R$ {stop_val:.2f} ({cfg.stop_loss_pct}%)\n"
            f"📈 EV mínimo: {cfg.min_ev_pct}%\n"
            f"🔄 Scan a cada: {cfg.scan_interval_min} min\n"
            f"⚽ Ligas: {', '.join(cfg.leagues)}\n\n"
            f"Vou analisar partidas automaticamente e enviar alertas quando encontrar oportunidades com vantagem matemática.\n\n"
            f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )
        return await self._send(msg)

    async def send_daily_summary(self, stats: Dict, banca: float) -> bool:
        sent    = stats.get("sent_today", 0)
        wins    = stats.get("wins", 0)
        losses  = stats.get("losses", 0)
        pending = sent - wins - losses
        roi     = stats.get("roi_today", 0.0)
        best_ev = stats.get("best_ev_today", 0.0)
        msg = (
            f"📊 <b>RESUMO DO DIA</b> — {datetime.now().strftime('%d/%m/%Y')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📤 Sinais enviados: {sent}\n"
            f"✅ Resultados +: {wins}\n"
            f"❌ Resultados -: {losses}\n"
            f"⏳ Pendentes: {pending}\n"
            f"📈 Melhor EV do dia: {best_ev:.1f}%\n"
            f"💰 ROI estimado: {roi:+.1f}%\n\n"
            f"Sistema continua monitorando 24h."
        )
        return await self._send(msg)

    async def send_no_opportunities(self) -> bool:
        msg = (
            f"🔍 Scan {datetime.now().strftime('%d/%m %H:%M')} — "
            f"Nenhuma oportunidade acima do limiar. Monitorando..."
        )
        return await self._send(msg)

    async def send_error_alert(self, error_msg: str) -> bool:
        msg = (
            f"⚠️ <b>ERRO NO SISTEMA</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<code>{error_msg[:400]}</code>\n"
            f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            f"O sistema vai tentar se recuperar no próximo ciclo."
        )
        return await self._send(msg)
