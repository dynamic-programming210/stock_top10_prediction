"""
E2: Logging and Monitoring Module
Provides structured logging, performance monitoring, and health checks
"""
import logging
import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUTS_DIR, MODEL_DIR, FEATURES_FILE, TOP10_LATEST_FILE

# Log directory
LOGS_DIR = OUTPUTS_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class StructuredLogger:
    """
    Structured logging with JSON output for monitoring
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler (human-readable)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (JSON structured)
        log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra={'data': kwargs})
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra={'data': kwargs})
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra={'data': kwargs})
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra={'data': kwargs})
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra={'data': kwargs})
    
    def log_event(self, event_type: str, details: Dict):
        """Log a structured event"""
        self.info(f"EVENT: {event_type}", event_type=event_type, **details)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'data') and record.data:
            log_entry['data'] = record.data
        
        return json.dumps(log_entry)


class PerformanceMonitor:
    """
    Track execution times and performance metrics
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger or StructuredLogger('performance')
        self.timings: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, any] = {}
    
    def start(self, stage: str):
        """Start timing a stage"""
        self.start_times[stage] = time.time()
        self.logger.info(f"Starting: {stage}")
    
    def stop(self, stage: str) -> float:
        """Stop timing a stage and return duration"""
        if stage not in self.start_times:
            self.logger.warning(f"Stage '{stage}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[stage]
        
        if stage not in self.timings:
            self.timings[stage] = []
        self.timings[stage].append(duration)
        
        self.logger.info(
            f"Completed: {stage}",
            stage=stage,
            duration_seconds=round(duration, 2)
        )
        
        del self.start_times[stage]
        return duration
    
    @contextmanager
    def measure(self, stage: str):
        """Context manager for timing a block"""
        self.start(stage)
        try:
            yield
        finally:
            self.stop(stage)
    
    def record_metric(self, name: str, value: any):
        """Record a custom metric"""
        self.metrics[name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"Metric: {name} = {value}")
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'stages': {},
            'metrics': self.metrics
        }
        
        for stage, times in self.timings.items():
            summary['stages'][stage] = {
                'count': len(times),
                'total_seconds': round(sum(times), 2),
                'avg_seconds': round(sum(times) / len(times), 2),
                'min_seconds': round(min(times), 2),
                'max_seconds': round(max(times), 2)
            }
        
        return summary
    
    def save_summary(self, filename: str = None):
        """Save performance summary to file"""
        if filename is None:
            filename = LOGS_DIR / f"perf_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = self.get_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Performance summary saved to {filename}")
        return str(filename)


def timed(func: Callable = None, stage_name: str = None):
    """Decorator to time function execution"""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            name = stage_name or fn.__name__
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                duration = time.time() - start
                logging.info(f"[TIMING] {name}: {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                logging.error(f"[TIMING] {name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


class HealthChecker:
    """
    System health checks for the stock prediction pipeline
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger or StructuredLogger('health')
        self.checks: List[Dict] = []
    
    def check_data_freshness(self, max_age_hours: int = 24) -> Dict:
        """Check if data files are recent"""
        check = {
            'name': 'data_freshness',
            'status': 'pass',
            'details': {}
        }
        
        files_to_check = {
            'features': FEATURES_FILE,
            'top10': TOP10_LATEST_FILE
        }
        
        now = datetime.now()
        
        for name, filepath in files_to_check.items():
            if not filepath.exists():
                check['status'] = 'fail'
                check['details'][name] = 'File not found'
                continue
            
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age_hours = (now - mtime).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                check['status'] = 'warn'
                check['details'][name] = f'Stale ({age_hours:.1f} hours old)'
            else:
                check['details'][name] = f'Fresh ({age_hours:.1f} hours old)'
        
        self.checks.append(check)
        return check
    
    def check_model_exists(self) -> Dict:
        """Check if trained models exist"""
        check = {
            'name': 'model_exists',
            'status': 'pass',
            'details': {}
        }
        
        model_files = list(MODEL_DIR.glob('*.joblib'))
        
        if not model_files:
            check['status'] = 'fail'
            check['details']['message'] = 'No trained models found'
        else:
            check['details']['models'] = [f.name for f in model_files]
            check['details']['count'] = len(model_files)
        
        self.checks.append(check)
        return check
    
    def check_data_coverage(self, min_symbols: int = 400) -> Dict:
        """Check feature data coverage"""
        check = {
            'name': 'data_coverage',
            'status': 'pass',
            'details': {}
        }
        
        try:
            import pandas as pd
            
            if not FEATURES_FILE.exists():
                check['status'] = 'fail'
                check['details']['message'] = 'Features file not found'
            else:
                df = pd.read_parquet(FEATURES_FILE)
                unique_symbols = df['symbol'].nunique() if 'symbol' in df.columns else 0
                
                check['details']['unique_symbols'] = unique_symbols
                check['details']['total_rows'] = len(df)
                
                if unique_symbols < min_symbols:
                    check['status'] = 'warn'
                    check['details']['warning'] = f'Low coverage: {unique_symbols}/{min_symbols} expected'
        
        except Exception as e:
            check['status'] = 'fail'
            check['details']['error'] = str(e)
        
        self.checks.append(check)
        return check
    
    def check_disk_space(self, min_gb: float = 1.0) -> Dict:
        """Check available disk space"""
        check = {
            'name': 'disk_space',
            'status': 'pass',
            'details': {}
        }
        
        try:
            import shutil
            total, used, free = shutil.disk_usage(OUTPUTS_DIR)
            
            free_gb = free / (1024 ** 3)
            check['details']['free_gb'] = round(free_gb, 2)
            check['details']['total_gb'] = round(total / (1024 ** 3), 2)
            
            if free_gb < min_gb:
                check['status'] = 'fail'
                check['details']['warning'] = f'Low disk space: {free_gb:.2f} GB'
        
        except Exception as e:
            check['status'] = 'warn'
            check['details']['error'] = str(e)
        
        self.checks.append(check)
        return check
    
    def run_all_checks(self) -> Dict:
        """Run all health checks"""
        self.checks = []
        
        self.check_data_freshness()
        self.check_model_exists()
        self.check_data_coverage()
        self.check_disk_space()
        
        overall_status = 'healthy'
        for check in self.checks:
            if check['status'] == 'fail':
                overall_status = 'unhealthy'
                break
            elif check['status'] == 'warn':
                overall_status = 'degraded'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'checks': self.checks
        }
        
        self.logger.log_event('health_check', report)
        
        return report
    
    def print_report(self, report: Dict = None):
        """Print health check report"""
        if report is None:
            report = self.run_all_checks()
        
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'unhealthy': '❌'
        }
        
        check_emoji = {
            'pass': '✅',
            'warn': '⚠️',
            'fail': '❌'
        }
        
        print("\n" + "=" * 50)
        print(f"🏥 SYSTEM HEALTH CHECK")
        print(f"   Timestamp: {report['timestamp']}")
        print(f"   Status: {status_emoji.get(report['overall_status'], '?')} {report['overall_status'].upper()}")
        print("=" * 50)
        
        for check in report['checks']:
            emoji = check_emoji.get(check['status'], '?')
            print(f"\n{emoji} {check['name']}")
            for key, value in check['details'].items():
                print(f"   • {key}: {value}")
        
        print("\n" + "=" * 50)


class AlertManager:
    """
    Simple alert manager for critical events
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger or StructuredLogger('alerts')
        self.alerts: List[Dict] = []
    
    def alert(self, level: str, message: str, **details):
        """Create an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details
        }
        
        self.alerts.append(alert)
        
        if level == 'critical':
            self.logger.critical(f"ALERT: {message}", **details)
        elif level == 'warning':
            self.logger.warning(f"ALERT: {message}", **details)
        else:
            self.logger.info(f"ALERT: {message}", **details)
    
    def critical(self, message: str, **details):
        self.alert('critical', message, **details)
    
    def warning(self, message: str, **details):
        self.alert('warning', message, **details)
    
    def info(self, message: str, **details):
        self.alert('info', message, **details)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) >= cutoff
        ]
    
    def save_alerts(self, filename: str = None):
        """Save alerts to file"""
        if filename is None:
            filename = LOGS_DIR / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.alerts, f, indent=2)
        
        return str(filename)


# Global instances for convenience
_default_logger = None
_default_monitor = None
_default_health = None
_default_alerts = None


def get_logger(name: str = 'stock_predictor') -> StructuredLogger:
    """Get or create default logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = StructuredLogger(name)
    return _default_logger


def get_monitor() -> PerformanceMonitor:
    """Get or create default performance monitor"""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = PerformanceMonitor()
    return _default_monitor


def get_health_checker() -> HealthChecker:
    """Get or create default health checker"""
    global _default_health
    if _default_health is None:
        _default_health = HealthChecker()
    return _default_health


def get_alert_manager() -> AlertManager:
    """Get or create default alert manager"""
    global _default_alerts
    if _default_alerts is None:
        _default_alerts = AlertManager()
    return _default_alerts


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("📊 STOCK PREDICTOR - HEALTH CHECK")
    print("=" * 50)
    
    health = HealthChecker()
    report = health.run_all_checks()
    health.print_report(report)
    
    print("\n📝 Testing Performance Monitor...")
    monitor = PerformanceMonitor()
    
    with monitor.measure("test_operation"):
        time.sleep(0.1)
    
    monitor.record_metric("test_metric", 42)
    summary = monitor.get_summary()
    print(f"   Performance summary: {json.dumps(summary, indent=2)}")
