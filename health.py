"""라즈베리파이 시스템 상태 수집. 비-RPi 환경에서는 가능한 항목만 채우고 나머지는 None."""
import json
import os
import subprocess

try:
    import psutil
except ImportError:
    psutil = None


def read_cpu_temp_c():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_mic_ok():
    # arecord -l 에 캡처 가능한 카드가 한 줄이라도 있으면 OK
    try:
        out = subprocess.check_output(
            ['arecord', '-l'], stderr=subprocess.STDOUT, timeout=2
        ).decode(errors='replace')
    except FileNotFoundError:
        return None
    except Exception:
        return False
    return any(line.startswith('card ') for line in out.splitlines())


def read_tailscale_ok():
    try:
        out = subprocess.check_output(
            ['tailscale', 'status', '--json'], stderr=subprocess.STDOUT, timeout=2
        ).decode(errors='replace')
    except FileNotFoundError:
        return None
    except Exception:
        return False
    try:
        return json.loads(out).get('BackendState') == 'Running'
    except Exception:
        return False


def read_throttled():
    try:
        out = subprocess.check_output(['vcgencmd', 'get_throttled'], timeout=2).decode().strip()
        # 예: "throttled=0x50000"
        hex_val = int(out.split('=')[1], 16)
    except Exception:
        return None
    return {
        'raw_hex': hex(hex_val),
        'under_voltage_now': bool(hex_val & 0x1),
        'freq_capped_now': bool(hex_val & 0x2),
        'throttled_now': bool(hex_val & 0x4),
        'soft_temp_limit_now': bool(hex_val & 0x8),
        'under_voltage_occurred': bool(hex_val & 0x10000),
        'freq_capped_occurred': bool(hex_val & 0x20000),
        'throttled_occurred': bool(hex_val & 0x40000),
        'soft_temp_limit_occurred': bool(hex_val & 0x80000),
    }


def get_health():
    info = {
        'cpu_temp_c': read_cpu_temp_c(),
        'throttled': read_throttled(),
        'cpu_percent': None,
        'memory_percent': None,
        'disk_percent': None,
        'load_avg_1m': None,
        'mic_ok': read_mic_ok(),
        'tailscale_ok': read_tailscale_ok(),
        'psutil_available': psutil is not None,
    }
    if psutil is not None:
        try:
            info['cpu_percent'] = psutil.cpu_percent(interval=None)
            info['memory_percent'] = psutil.virtual_memory().percent
            info['disk_percent'] = psutil.disk_usage('/').percent
        except Exception:
            pass
    try:
        info['load_avg_1m'] = os.getloadavg()[0]
    except (AttributeError, OSError):
        pass
    return info
