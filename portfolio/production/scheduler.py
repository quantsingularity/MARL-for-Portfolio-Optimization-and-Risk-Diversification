"""
Production Scheduler for Portfolio Rebalancing
Handles scheduled portfolio rebalancing at configurable frequencies
"""

import json
import logging
import os
import sys

import requests

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError as _exc:  # pragma: no cover - optional production dependency
    BlockingScheduler = None
    CronTrigger = None
    IntervalTrigger = None
    _APSCHEDULER_IMPORT_ERROR = _exc
else:
    _APSCHEDULER_IMPORT_ERROR = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging. Log directory is configurable via LOG_DIR and created if
# missing so the module imports cleanly outside the Docker container.
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "scheduler.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RebalancingScheduler:
    """Scheduler for automated portfolio rebalancing"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        if BlockingScheduler is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "apscheduler is required for the rebalancing scheduler. "
                "Install it with `pip install apscheduler`."
            ) from _APSCHEDULER_IMPORT_ERROR
        self.api_url = api_url
        self.scheduler = BlockingScheduler()
        self.portfolios = {}

    def load_portfolio_schedules(
        self, config_path: str = "/app/configs/schedules.json"
    ):
        """Load portfolio rebalancing schedules from config"""
        try:
            with open(config_path, "r") as f:
                self.portfolios = json.load(f)
            logger.info(f"Loaded {len(self.portfolios)} portfolio schedules")
        except FileNotFoundError:
            logger.warning(f"Schedule config not found: {config_path}")
            self.portfolios = {}

    def rebalance_portfolio(self, portfolio_id: str):
        """Execute portfolio rebalancing"""
        logger.info(f"Starting rebalancing for portfolio: {portfolio_id}")

        try:
            # Call API to trigger rebalancing
            response = requests.post(
                f"{self.api_url}/portfolio/{portfolio_id}/rebalance", timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Successfully triggered rebalancing for {portfolio_id}")
                return True
            else:
                logger.error(f"Rebalancing failed for {portfolio_id}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error rebalancing {portfolio_id}: {str(e)}")
            return False

    def daily_rebalance_job(self, portfolio_id: str):
        """Daily rebalancing job"""
        logger.info(f"[DAILY] Rebalancing {portfolio_id}")
        self.rebalance_portfolio(portfolio_id)

    def weekly_rebalance_job(self, portfolio_id: str):
        """Weekly rebalancing job"""
        logger.info(f"[WEEKLY] Rebalancing {portfolio_id}")
        self.rebalance_portfolio(portfolio_id)

    def monthly_rebalance_job(self, portfolio_id: str):
        """Monthly rebalancing job"""
        logger.info(f"[MONTHLY] Rebalancing {portfolio_id}")
        self.rebalance_portfolio(portfolio_id)

    def setup_schedules(self):
        """Setup rebalancing schedules based on configuration"""
        for portfolio_id, config in self.portfolios.items():
            frequency = config.get("frequency", "weekly")

            if frequency == "daily":
                # Daily at market close (4:00 PM ET)
                self.scheduler.add_job(
                    self.daily_rebalance_job,
                    CronTrigger(hour=16, minute=0),
                    args=[portfolio_id],
                    id=f"daily_{portfolio_id}",
                    replace_existing=True,
                )
                logger.info(f"Scheduled daily rebalancing for {portfolio_id}")

            elif frequency == "weekly":
                # Weekly on Friday at market close
                self.scheduler.add_job(
                    self.weekly_rebalance_job,
                    CronTrigger(day_of_week="fri", hour=16, minute=0),
                    args=[portfolio_id],
                    id=f"weekly_{portfolio_id}",
                    replace_existing=True,
                )
                logger.info(f"Scheduled weekly rebalancing for {portfolio_id}")

            elif frequency == "monthly":
                # Monthly on first trading day
                self.scheduler.add_job(
                    self.monthly_rebalance_job,
                    CronTrigger(day=1, hour=16, minute=0),
                    args=[portfolio_id],
                    id=f"monthly_{portfolio_id}",
                    replace_existing=True,
                )
                logger.info(f"Scheduled monthly rebalancing for {portfolio_id}")

    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.debug("API health check: OK")
                return True
            else:
                logger.warning(f"API health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API health check error: {str(e)}")
            return False

    def start(self):
        """Start the scheduler"""
        logger.info("Starting rebalancing scheduler...")

        # Load schedules
        self.load_portfolio_schedules()

        # Setup jobs
        self.setup_schedules()

        # Add health check job (every 5 minutes)
        self.scheduler.add_job(
            self.health_check,
            IntervalTrigger(minutes=5),
            id="health_check",
            replace_existing=True,
        )

        # Start scheduler
        logger.info("Scheduler started successfully")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")


def main():
    """Main entry point"""
    api_url = os.getenv("API_URL", "http://api:8000")

    scheduler = RebalancingScheduler(api_url)
    scheduler.start()


if __name__ == "__main__":
    main()
