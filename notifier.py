"""
Notifier — Envia alertas formatados pelo Telegram
===================================================
Usa a API oficial do Telegram (Bot API) — gratuita e sem restrições.
"""

import logging
import aiohttp
from datetime import datetime
from typing import Dict, Optional
from config import Config

log = logging.getLogger(__name__)

BETANO_URL = "https://www.betano.com.br/sport/futebol/"

# Navegação Betano por mercado
BETANO_PATH = {
    "home":    ("Principais", "Resultado Final", "1 — Vitória Casa"),
    "draw":    ("Principais", "Resultado Final", "X — Empate"),
    "away":    ("Principais", "Resultado Final", "2 — Vitória Visitante"),
    "over25":  ("Gols",       "Total de Gols",   "Mais de 2.5"),
    "under25": ("Gols",       "Total de Gols",   "Menos de 2.5"),
    "btts":    ("Gols",       "Ambas Marcam",    "Sim"),
}


class Notifier:
    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.base    = f"https://api.telegram.org/bot{token}"

    async def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Envia uma mensagem pelo Telegram."""
        url = f"{self.base}/sendMessage"
        payload = {
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    ok = r.status == 200
                    if not ok:
                        body = await r.text()
                        log.error(f"Telegram API erro {r.status}: {body}")
                    return ok
        except Exception as e:
            log.error(f"Falha ao enviar mensagem Telegram: {e}")
            return False

    async def send_opportunity(self, opp: Dict, banca: float) -> bool:
        """Formata e envia um alerta de oportunidade."""
        nav = BETANO_PATH.get(opp["market_key"], ("Principais", opp["market_label"], opp["market_label"]))
        tab, section, option = nav

        retorno = opp["stake"] * opp["odds"]
        lucro   = opp["stake"] * (opp["odds"] - 1)
        odd_min = round(opp["odds"] * 0.97, 2)

        # Linha de mercados alternativos com EV+
        alt_markets = [
            m for m in opp.get("all_markets", [])
            if m["ev_pct"] > 0 and m["label"] != opp["market_label"]
        ][:2]
        alt_text = ""
        if alt_markets:
            alt_text = "\n" + "\n".join(
                f"   • {m['label']} @ {m['odds']:.2f} — EV {m['ev_pct']:.1f}%"
                for m in alt_markets
            )

        msg = f"""🎯 <b>OPORTUNIDADE DETECTADA</b>
━━━━━━━━━━━━━━━━━━━━━━━━

⚽ <b>{opp['home_team']} vs {opp['away_team']}</b>
📅 {opp['date']} | {opp['league']}

📊 <b>MERCADO:</b> {opp['market_label']}
💰 <b>ODD:</b> {opp['odds']:.2f}  |  Mínima: {odd_min}
📈 <b>EV:</b> +{opp['ev_pct']:.1f}%  |  Edge: {opp['edge_pct']:.1f}%
🏦 <b>APOSTAR:</b> R$ {opp['stake']:.2f}
💵 Retorno: R$ {retorno:.2f}  |  Lucro: R$ {lucro:.2f}
{opp['cls_color']} Classificação: {opp['cls_label']}

📍 <b>COMO APOSTAR NA BETANO:</b>
<code>Futebol → {opp['league']}
→ {opp['home_team']} vs {opp['away_team']}
→ Aba: {tab} → {section}
→ Selecionar: {option}</code>

⚠️ Odd mínima aceitável: <b>{odd_min}</b>
⏱ Execute em até 5 minutos

<b>Outros mercados com valor:</b>{alt_text if alt_text else "\n   — Apenas este mercado tem EV positivo"}

━━━━━━━━━━━━━━━━━━━━━━━━
🧮 Prob real: {opp['real_prob']*100:.1f}%  |  Impl: {opp['implied_prob']*100:.1f}%
📊 Kelly: {opp['kelly_frac']*100:.2f}% da banca
🕐 Detectado: {datetime.now().strftime('%H:%M:%S')}"""

        return await self._send(msg)

    async def send_ai_analysis(self, opp: Dict) -> bool:
        """Envia a análise qualitativa separada para não poluir o alerta principal."""
        if not opp.get("ai_analysis"):
            return False
        msg = f"""🤖 <b>ANÁLISE DA IA</b> — {opp['home_team']} vs {opp['away_team']}
━━━━━━━━━━━━━━━━━━━━━━━━

{opp['ai_analysis']}"""
        return await self._send(msg)

    async def send_startup_message(self, cfg: Config) -> bool:
        msg = f"""🟢 <b>BETTING AI ENGINE — ONLINE</b>
━━━━━━━━━━━━━━━━━━━━━━━━

⚙️ <b>Configuração ativa:</b>
💰 Banca: R$ {cfg.banca:.2f}
🛑 Stop loss: R$ {cfg.banca * cfg.stop_loss_pct / 100:.2f} ({cfg.stop_loss_pct}%)
📈 EV mínimo: {cfg.min_ev_pct}%
🔄 Scan a cada: {cfg.scan_interval_min} min ({cfg.scan_interval_min//60}h)
⚽ Ligas: {', '.join(cfg.leagues)}

O sistema está analisando partidas e enviará alertas automaticamente quando encontrar oportunidades com vantagem matemática.

🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"""
        return await self._send(msg)

    async def send_daily_summary(self, stats: Dict, banca: float) -> bool:
        """Resumo diário dos sinais enviados."""
        sent    = stats.get("sent_today", 0)
        wins    = stats.get("wins", 0)
        losses  = stats.get("losses", 0)
        pending = sent - wins - losses
        roi     = stats.get("roi_today", 0.0)
        best    = stats.get("best_ev_today", 0.0)

        msg = f"""📊 <b>RESUMO DO DIA</b> — {datetime.now().strftime('%d/%m/%Y')}
━━━━━━━━━━━━━━━━━━━━━━━━

📤 Sinais enviados: {sent}
✅ Resultados +: {wins}
❌ Resultados -: {losses}
⏳ Pendentes: {pending}
📈 Melhor EV do dia: {best:.1f}%

💰 ROI estimado hoje: {roi:+.1f}%

Sistema continua monitorando 24h.
Próximo scan em {stats.get('next_scan_in', '?')} min."""
        return await self._send(msg)

    async def send_error_alert(self, error_msg: str) -> bool:
        msg = f"""⚠️ <b>ERRO NO SISTEMA</b>
━━━━━━━━━━━━━━━━━━━━━━━━
{error_msg[:500]}
🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"""
        return await self._send(msg)

    async def send_no_opportunities(self) -> bool:
        msg = f"🔍 Scan {datetime.now().strftime('%H:%M')} — Nenhuma oportunidade acima do limiar. Sistema continua monitorando."
        return await self._send(msg)
