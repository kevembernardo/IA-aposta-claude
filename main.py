"""
BETTING AI ENGINE — Sistema Autônomo 24/7
==========================================
Roda continuamente, analisa partidas, calcula EV/Kelly
e envia alertas pelo Telegram.
"""

import asyncio
import logging
import os
import sys

# ── CRIAR PASTAS ANTES DE QUALQUER COISA (fix crash Railway) ────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── CARREGAR .env SE EXISTIR (ambiente local) ────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # No Railway as variáveis já vêm do dashboard, não precisa de dotenv

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
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),   # Railway captura stdout
    ]
)
log = logging.getLogger(__name__)


async def run_cycle(scanner: Scanner, notifier: Notifier, memory: Memory, cfg: Config):
    """Executa um ciclo completo de análise."""
    log.info("=" * 50)
    log.info(f"Iniciando ciclo — {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    try:
        # 1. Buscar partidas e calcular oportunidades (passa banca e kelly)
        opportunities = await scanner.scan(
            leagues=cfg.leagues,
            max_matches=cfg.max_matches,
            banca=cfg.banca,
            kelly_fraction=cfg.kelly_fraction,
        )
        log.info(f"{len(opportunities)} oportunidades com EV+ encontradas")

        if not opportunities:
            log.info("Sem oportunidades. Aguardando próximo ciclo.")
            await notifier.send_no_opportunities()
            return

        # 2. Filtrar pelo EV mínimo (adaptativo baseado no histórico)
        min_ev = memory.get_adaptive_min_ev(cfg.min_ev_pct)
        filtered = [o for o in opportunities if o["ev_pct"] >= min_ev]
        log.info(f"{len(filtered)} acima do EV mínimo ({min_ev:.1f}%)")

        # 3. Enviar alertas
        sent = 0
        for opp in filtered:
            match_key = f"{opp['home_team']}-{opp['away_team']}"

            if memory.already_sent_today(match_key, opp["market_key"]):
                log.info(f"Já enviado hoje: {match_key} — {opp['market_key']}")
                continue

            success = await notifier.send_opportunity(opp, cfg.banca)
            if success:
                # Análise detalhada em mensagem separada
                await asyncio.sleep(1)
                await notifier.send_ai_analysis(opp)
                memory.record_sent(match_key, opp)
                sent += 1
                await asyncio.sleep(2)

        log.info(f"{sent} alerta(s) enviado(s) via Telegram")

        # 4. Resumo diário às 8h
        now = datetime.now()
        if now.hour == cfg.summary_hour and now.minute < 10:
            stats = memory.get_daily_stats()
            await notifier.send_daily_summary(stats, cfg.banca)

    except Exception as e:
        log.error(f"Erro no ciclo: {e}", exc_info=True)
        try:
            await notifier.send_error_alert(str(e))
        except Exception:
            pass  # Não crashar se o próprio envio de erro falhar


async def main():
    log.info("=" * 60)
    log.info("🤖 BETTING AI ENGINE — INICIANDO")
    log.info("=" * 60)

    # Validar configuração antes de começar
    cfg = Config()
    try:
        cfg.validate()
    except ValueError as e:
        log.error(f"CONFIGURAÇÃO INVÁLIDA:\n{e}")
        log.error("Verifique as variáveis de ambiente no Railway e reinicie.")
        sys.exit(1)

    log.info(f"Banca: R$ {cfg.banca:.2f}")
    log.info(f"EV mínimo: {cfg.min_ev_pct}%")
    log.info(f"Scan a cada: {cfg.scan_interval_min} min")
    log.info(f"Ligas: {', '.join(cfg.leagues)}")

    memory   = Memory("data/history.json")
    notifier = Notifier(cfg.telegram_token, cfg.telegram_chat_id)
    scanner  = Scanner(cfg.anthropic_api_key)

    # Mensagem de startup — se falhar aqui, o Telegram está mal configurado
    log.info("Enviando mensagem de startup para o Telegram...")
    ok = await notifier.send_startup_message(cfg)
    if not ok:
        log.error("FALHA ao enviar mensagem no Telegram!")
        log.error("Verifique TELEGRAM_TOKEN e TELEGRAM_CHAT_ID.")
        log.error("Continuando mesmo assim — verificar logs para erros de Telegram.")

    log.info("Sistema ativo. Iniciando primeiro ciclo...")

    cycle = 0
    while True:
        cycle += 1
        log.info(f"─── Ciclo #{cycle} ───")
        await run_cycle(scanner, notifier, memory, cfg)
        log.info(f"Próximo ciclo em {cfg.scan_interval_min} minutos...")
        await asyncio.sleep(cfg.scan_interval_min * 60)


if __name__ == "__main__":
    asyncio.run(main())
