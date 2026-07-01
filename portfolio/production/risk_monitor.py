"""
Risk Monitoring Service
Continuously monitors portfolio risk metrics and sends alerts
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging. The log directory is configurable via the LOG_DIR
# environment variable and defaults to a local ``logs/`` directory so the module
# imports cleanly outside the Docker container (where /app/logs does not exist).
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "risk_monitor.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RiskAlert:
    """Risk alert definition"""

    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        threshold: float,
        current_value: float,
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.threshold = threshold
        self.current_value = current_value
        self.timestamp = datetime.now()

    def to_dict(self):
        return {
            "type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskMonitor:
    """Real-time risk monitoring service"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.portfolios = {}
        self.alerts = []

        # Risk thresholds
        self.thresholds = {
            "max_drawdown": 0.15,
            "var_95": 0.025,
            "volatility": 0.25,
            "concentration": 0.30,  # Single asset weight
            "correlation": 0.70,  # Avg pairwise correlation
        }

        # Alert severities
        self.severities = {"info": 0, "warning": 1, "critical": 2}

    def load_portfolios(self, config_path: str = "/app/configs/risk_config.json"):
        """Load portfolios to monitor"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                self.portfolios = config.get("portfolios", [])
                self.thresholds.update(config.get("thresholds", {}))
            logger.info(f"Loaded {len(self.portfolios)} portfolios to monitor")
        except FileNotFoundError:
            logger.warning(f"Risk config not found: {config_path}")
            # Try to get portfolios from API
            self.fetch_active_portfolios()

    def fetch_active_portfolios(self):
        """Fetch active portfolios from API"""
        try:
            response = requests.get(f"{self.api_url}/portfolio/list", timeout=10)
            if response.status_code == 200:
                self.portfolios = response.json().get("portfolios", [])
                logger.info(f"Fetched {len(self.portfolios)} active portfolios")
        except Exception as e:
            logger.error(f"Error fetching portfolios: {str(e)}")

    def check_drawdown(self, portfolio_id: str, metrics: Dict) -> Optional[RiskAlert]:
        """Check for excessive drawdown"""
        current_dd = metrics.get("max_drawdown", 0)
        threshold = self.thresholds["max_drawdown"]

        if current_dd > threshold * 0.8:  # 80% of threshold
            severity = "warning" if current_dd < threshold else "critical"
            return RiskAlert(
                alert_type="drawdown",
                severity=severity,
                message=f"Portfolio drawdown at {current_dd*100:.2f}% (threshold: {threshold*100:.0f}%)",
                threshold=threshold,
                current_value=current_dd,
            )
        return None

    def check_volatility(self, portfolio_id: str, metrics: Dict) -> Optional[RiskAlert]:
        """Check for excessive volatility"""
        current_vol = metrics.get("volatility", 0)
        threshold = self.thresholds["volatility"]

        if current_vol > threshold * 0.8:
            severity = "warning" if current_vol < threshold else "critical"
            return RiskAlert(
                alert_type="volatility",
                severity=severity,
                message=f"Portfolio volatility at {current_vol*100:.2f}% (threshold: {threshold*100:.0f}%)",
                threshold=threshold,
                current_value=current_vol,
            )
        return None

    def check_concentration(
        self, portfolio_id: str, allocations: List[Dict]
    ) -> Optional[RiskAlert]:
        """Check for concentration risk"""
        if not allocations:
            return None

        max_weight = max([a.get("weight", 0) for a in allocations])
        threshold = self.thresholds["concentration"]

        if max_weight > threshold * 0.8:
            severity = "warning" if max_weight < threshold else "critical"
            ticker = [
                a["ticker"] for a in allocations if a.get("weight") == max_weight
            ][0]
            return RiskAlert(
                alert_type="concentration",
                severity=severity,
                message=f"High concentration in {ticker}: {max_weight*100:.2f}% (threshold: {threshold*100:.0f}%)",
                threshold=threshold,
                current_value=max_weight,
            )
        return None

    def check_var(self, portfolio_id: str, metrics: Dict) -> Optional[RiskAlert]:
        """Check Value at Risk"""
        current_var = metrics.get("var_95", 0)
        threshold = self.thresholds["var_95"]

        if current_var > threshold * 0.8:
            severity = "warning" if current_var < threshold else "critical"
            return RiskAlert(
                alert_type="var",
                severity=severity,
                message=f"VaR(95%) at {current_var*100:.2f}% (threshold: {threshold*100:.0f}%)",
                threshold=threshold,
                current_value=current_var,
            )
        return None

    def monitor_portfolio(self, portfolio_id: str) -> List[RiskAlert]:
        """Monitor a single portfolio and generate alerts"""
        alerts = []

        try:
            # Get risk metrics
            response = requests.get(
                f"{self.api_url}/risk/metrics/{portfolio_id}", timeout=10
            )

            if response.status_code == 200:
                metrics = response.json()

                # Check various risk factors
                alert = self.check_drawdown(portfolio_id, metrics)
                if alert:
                    alerts.append(alert)

                alert = self.check_volatility(portfolio_id, metrics)
                if alert:
                    alerts.append(alert)

                alert = self.check_var(portfolio_id, metrics)
                if alert:
                    alerts.append(alert)

            # Get portfolio allocations
            response = requests.get(
                f"{self.api_url}/portfolio/{portfolio_id}", timeout=10
            )

            if response.status_code == 200:
                portfolio = response.json()
                allocations = portfolio.get("allocations", [])

                alert = self.check_concentration(portfolio_id, allocations)
                if alert:
                    alerts.append(alert)

        except Exception as e:
            logger.error(f"Error monitoring {portfolio_id}: {str(e)}")

        return alerts

    def send_alerts(self, portfolio_id: str, alerts: List[RiskAlert]):
        """Send risk alerts (email, slack, etc.)"""
        if not alerts:
            return

        critical_alerts = [a for a in alerts if a.severity == "critical"]
        warning_alerts = [a for a in alerts if a.severity == "warning"]

        logger.info(
            f"Portfolio {portfolio_id}: {len(critical_alerts)} critical, {len(warning_alerts)} warnings"
        )

        for alert in alerts:
            logger.warning(
                f"[{alert.severity.upper()}] {portfolio_id}: {alert.message}"
            )

            # Store alert
            self.alerts.append({"portfolio_id": portfolio_id, "alert": alert.to_dict()})

        # In production, send to notification channels:
        # - Email
        # - Slack/Teams
        # - SMS for critical alerts
        # - Dashboard notifications

    def save_alerts_log(self, output_path: str = None):
        """Save alerts to log file"""
        if output_path is None:
            output_path = os.path.join(LOG_DIR, "risk_alerts.json")
        try:
            with open(output_path, "w") as f:
                json.dump(self.alerts[-1000:], f, indent=2)  # Keep last 1000 alerts
        except Exception as e:
            logger.error(f"Error saving alerts log: {str(e)}")

    def generate_risk_report(self, output_dir: str = None):
        """Generate daily risk report"""
        if output_dir is None:
            output_dir = os.environ.get(
                "REPORT_DIR", os.path.join(os.getcwd(), "reports")
            )
        os.makedirs(output_dir, exist_ok=True)

        report_date = datetime.now().strftime("%Y-%m-%d")
        report_path = f"{output_dir}/risk_report_{report_date}.json"

        # Aggregate statistics
        recent_alerts = [
            a
            for a in self.alerts
            if (datetime.now() - datetime.fromisoformat(a["alert"]["timestamp"])).days
            < 1
        ]

        critical_count = sum(
            1 for a in recent_alerts if a["alert"]["severity"] == "critical"
        )
        warning_count = sum(
            1 for a in recent_alerts if a["alert"]["severity"] == "warning"
        )

        report = {
            "date": report_date,
            "summary": {
                "portfolios_monitored": len(self.portfolios),
                "total_alerts": len(recent_alerts),
                "critical_alerts": critical_count,
                "warning_alerts": warning_count,
            },
            "alerts_by_type": {},
            "alerts_by_portfolio": {},
            "recent_alerts": recent_alerts[-50:],  # Last 50 alerts
        }

        # Group by type
        for alert_data in recent_alerts:
            alert = alert_data["alert"]
            alert_type = alert["type"]
            if alert_type not in report["alerts_by_type"]:
                report["alerts_by_type"][alert_type] = 0
            report["alerts_by_type"][alert_type] += 1

        # Group by portfolio
        for alert_data in recent_alerts:
            portfolio_id = alert_data["portfolio_id"]
            if portfolio_id not in report["alerts_by_portfolio"]:
                report["alerts_by_portfolio"][portfolio_id] = 0
            report["alerts_by_portfolio"][portfolio_id] += 1

        # Save report
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Risk report generated: {report_path}")

    def run(self, interval: int = 300):
        """Run continuous monitoring"""
        logger.info("Starting risk monitoring service...")

        # Load portfolios
        self.load_portfolios()

        iteration = 0

        try:
            while True:
                logger.info(f"Monitoring iteration {iteration + 1}")

                # Monitor each portfolio
                for portfolio_id in self.portfolios:
                    alerts = self.monitor_portfolio(portfolio_id)
                    if alerts:
                        self.send_alerts(portfolio_id, alerts)

                # Save alerts log
                self.save_alerts_log()

                # Generate daily report (at midnight)
                current_hour = datetime.now().hour
                if current_hour == 0 and datetime.now().minute < 10:
                    self.generate_risk_report()

                iteration += 1

                # Wait before next iteration
                logger.info(f"Waiting {interval} seconds until next check...")
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Risk monitoring stopped by user")
        except Exception as e:
            logger.error(f"Risk monitoring error: {str(e)}")


def main():
    """Main entry point"""
    api_url = os.getenv("API_URL", "http://api:8000")
    interval = int(os.getenv("MONITOR_INTERVAL", "300"))  # 5 minutes default

    monitor = RiskMonitor(api_url)
    monitor.run(interval)


if __name__ == "__main__":
    main()
