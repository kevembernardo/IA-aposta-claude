"""
BETTING AI ENGINE — Sistema Autônomo 24/7
==========================================
Roda continuamente, analisa partidas, calcula EV/Kelly
e envia alertas pelo Telegram.
"""

import asyncio
import logging
import os
from datetime import datetime
from engine.scanner import Scanner
from engine.notifier import Notifier
from engine.memory import Memory
from config import Config

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


async def run_cycle(scanner: Scanner, notifier: Notifier, memory: Memory, cfg: Config):
    """Executa um ciclo completo de análise."""
    log.info("=" * 50)
    log.info(f"Iniciando ciclo de análise — {datetime.now().strftime('%d/%m %H:%M')}")

    try:
        # 1. Buscar e analisar partidas
        opportunities = await scanner.scan(cfg.leagues, cfg.max_matches)
        log.info(f"{len(opportunities)} oportunidades encontradas")

        if not opportunities:
            log.info("Nenhuma oportunidade acima do limiar. Aguardando próximo ciclo.")
            return

        # 2. Filtrar pelo EV mínimo adaptativo (aprende com histórico)
        min_ev = memory.get_adaptive_min_ev(cfg.min_ev_pct)
        filtered = [o for o in opportunities if o["ev_pct"] >= min_ev]
        log.info(f"{len(filtered)} oportunidades acima do EV mínimo ({min_ev:.1f}%)")

        # 3. Enviar alertas
        sent = 0
        for opp in filtered:
            match_key = f"{opp['home_team']}-{opp['away_team']}"

            # Evitar reenviar a mesma oportunidade no mesmo dia
            if memory.already_sent_today(match_key, opp["market_key"]):
                log.info(f"Pulando (já enviado hoje): {match_key} — {opp['market_key']}")
                continue

            await notifier.send_opportunity(opp, cfg.banca)
            memory.record_sent(match_key, opp)
            sent += 1
            await asyncio.sleep(1)  # throttle

        log.info(f"{sent} alertas enviados via Telegram")

        # 4. Resumo diário (se for hora)
        if datetime.now().hour == cfg.summary_hour and datetime.now().minute < cfg.scan_interval_min:
            stats = memory.get_daily_stats()
            await notifier.send_daily_summary(stats, cfg.banca)

    except Exception as e:
        log.error(f"Erro no ciclo: {e}", exc_info=True)
        await notifier.send_error_alert(str(e))


async def main():
    log.info("🤖 BETTING AI ENGINE iniciando...")

    cfg      = Config()
    memory   = Memory("data/history.json")
    notifier = Notifier(cfg.telegram_token, cfg.telegram_chat_id)
    scanner  = Scanner(cfg.anthropic_api_key)

    await notifier.send_startup_message(cfg)
    log.info(f"Sistema ativo. Scan a cada {cfg.scan_interval_min} minutos.")
    log.info(f"Ligas: {', '.join(cfg.leagues)}")

    while True:
        await run_cycle(scanner, notifier, memory, cfg)
        log.info(f"Próximo scan em {cfg.scan_interval_min} minutos...")
        await asyncio.sleep(cfg.scan_interval_min * 60)


if __name__ == "__main__":
    asyncio.run(main())
