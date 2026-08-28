"""The Canvas load readout: four numbers, and the honesty to omit them.

This endpoint is polled by every open tab, so the two things worth pinning are
not the happy path. They are: a machine WITHOUT an NVIDIA card must answer
without the GPU keys (never a zero the widget would draw as "the GPU is idle"),
and N pollers must cost ONE `nvidia-smi`, not N.
"""
import subprocess
from unittest.mock import patch

import pytest

from app.services import system_stats


class _Proc:
    def __init__(self, stdout='', returncode=0):
        self.stdout = stdout
        self.returncode = returncode


@pytest.fixture(autouse=True)
def _fresh_cache():
    system_stats.reset_cache()
    yield
    system_stats.reset_cache()


def test_a_machine_with_a_gpu_reports_util_vram_and_temperature():
    with patch.object(subprocess, 'run',
                      return_value=_Proc('87, 21504, 24564, 47\n')) as run:
        data = system_stats.machine_stats()
    assert run.call_args[0][0][0] == 'nvidia-smi'
    assert 'temperature.gpu' in run.call_args[0][0][1]
    assert data['gpu_percent'] == 87
    # MiB -> GB, one decimal: 21504/1024 = 21.0, 24564/1024 = 23.99 -> 24.0
    assert data['vram_used_gb'] == 21.0
    assert data['vram_total_gb'] == 24.0
    assert data['gpu_temp_c'] == 47
    # psutil is a hard requirement of the app, so CPU/RAM come for free.
    assert 0 <= data['cpu_percent'] <= 100
    assert data['ram_total_gb'] > 0
    assert data['ram_used_gb'] <= data['ram_total_gb']


def test_a_gpu_that_cannot_report_temperature_keeps_its_other_numbers():
    """"[N/A]" for temperature (virtualised and some laptop GPUs) costs the one
    number it is — never the whole GPU sample."""
    with patch.object(subprocess, 'run',
                      return_value=_Proc('87, 21504, 24564, [N/A]\n')):
        data = system_stats.machine_stats()
    assert data['gpu_percent'] == 87
    assert data['vram_used_gb'] == 21.0
    assert 'gpu_temp_c' not in data


def test_no_nvidia_card_drops_the_gpu_keys_instead_of_answering_zero():
    """The whole point of omission: `gpu_percent: 0` on a GPU-less machine
    would be drawn as an idle card that is not there."""
    with patch.object(subprocess, 'run', side_effect=OSError('nvidia-smi not found')):
        data = system_stats.machine_stats()
    assert 'gpu_percent' not in data
    assert 'vram_used_gb' not in data
    assert 'vram_total_gb' not in data
    assert 'gpu_temp_c' not in data
    # …and the CPU/RAM half still answers. A missing GPU is not an outage.
    assert 'cpu_percent' in data and 'ram_total_gb' in data


@pytest.mark.parametrize('proc, why', [
    (_Proc('', returncode=9), 'nvidia-smi failed'),
    (_Proc('\n'), 'empty output'),
    (_Proc('87, 21504\n'), 'truncated row'),
    (_Proc('[N/A], 21504, 24564\n'), 'utilization unsupported on this GPU'),
])
def test_every_unusable_nvidia_smi_answer_is_treated_as_unknown(proc, why):
    with patch.object(subprocess, 'run', return_value=proc):
        data = system_stats.machine_stats()
    assert 'gpu_percent' not in data, why


def test_nvidia_smi_timing_out_does_not_take_the_endpoint_down():
    with patch.object(subprocess, 'run',
                      side_effect=subprocess.TimeoutExpired('nvidia-smi', 5)):
        data = system_stats.machine_stats()
    assert 'gpu_percent' not in data
    assert 'cpu_percent' in data


def test_a_burst_of_clients_costs_exactly_one_probe():
    """Ten tabs polling at 5 s must not fork nvidia-smi ten times."""
    with patch.object(subprocess, 'run',
                      return_value=_Proc('40, 4096, 24564\n')) as run:
        seen = [system_stats.machine_stats() for _ in range(10)]
    assert run.call_count == 1
    assert all(s['gpu_percent'] == 40 for s in seen)


def test_the_cached_reading_cannot_be_mutated_by_one_client():
    with patch.object(subprocess, 'run', return_value=_Proc('40, 4096, 24564\n')):
        first = system_stats.machine_stats()
        first['gpu_percent'] = 999
        second = system_stats.machine_stats()
    assert second['gpu_percent'] == 40


class _Times(tuple):
    """Stand-in for psutil's scputimes namedtuple."""
    _fields = ('user', 'system', 'idle')


class _FakePsutil:
    """A psutil whose cpu_percent() is THREAD-KEYED, like the real one.

    This is the trap the service exists to dodge: psutil stores the previous
    cpu_times per `threading.current_thread().ident`, so on a threaded web
    server nearly every request is that worker's first call and gets 0.0.
    """

    def __init__(self, samples):
        self._samples = list(samples)
        self._seen_threads = set()
        self.blocking_calls = 0

    def cpu_times(self):
        return self._samples.pop(0) if self._samples else _Times((0, 0, 0))

    def cpu_percent(self, interval=None):
        import threading
        if interval is not None:
            self.blocking_calls += 1
            return 71.0
        ident = threading.current_thread().ident
        if ident not in self._seen_threads:
            self._seen_threads.add(ident)
            return 0.0     # the silent failure, reproduced
        return 55.0

    def virtual_memory(self):
        raise RuntimeError('not under test here')


def test_cpu_is_measured_from_our_own_delta_not_psutils_thread_keyed_one():
    """The bug this pins: served from a Flask worker pool, cpu_percent(None)
    answered a flat 0 forever on a machine sitting at 20-30%. The service keeps
    the previous cpu_times itself, so the reading survives changing threads."""
    ps = _FakePsutil([
        _Times((100.0, 50.0, 850.0)),     # baseline
        _Times((130.0, 60.0, 1010.0)),    # +30 user, +10 sys, +160 idle -> 20%
    ])
    system_stats.reset_cache(forget_cpu=True)
    with patch.object(system_stats, '_psutil', return_value=ps), \
         patch.object(subprocess, 'run', side_effect=OSError('no nvidia-smi')):
        first = system_stats.machine_stats()          # cold: one blocking sample
        system_stats.reset_cache()
        second = system_stats.machine_stats()         # warm: our own delta
    assert first['cpu_percent'] == 71                 # measured, not a flat 0
    assert ps.blocking_calls == 1                     # …and only once, ever
    assert second['cpu_percent'] == 20


def test_the_cpu_reading_survives_being_served_from_a_different_thread():
    import threading
    ps = _FakePsutil([
        _Times((0.0, 0.0, 0.0)),
        _Times((10.0, 0.0, 90.0)),
        _Times((60.0, 0.0, 140.0)),       # +50 busy / +100 total -> 50%
    ])
    system_stats.reset_cache(forget_cpu=True)
    out = {}
    with patch.object(system_stats, '_psutil', return_value=ps), \
         patch.object(subprocess, 'run', side_effect=OSError('no nvidia-smi')):
        system_stats.machine_stats()
        system_stats.reset_cache()
        system_stats.machine_stats()
        system_stats.reset_cache()

        def worker():
            out['data'] = system_stats.machine_stats()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert out['data']['cpu_percent'] == 50


def test_a_zero_length_window_reads_as_zero_rather_than_dividing_by_it():
    ps = _FakePsutil([_Times((5.0, 5.0, 5.0)), _Times((5.0, 5.0, 5.0))])
    system_stats.reset_cache(forget_cpu=True)
    with patch.object(system_stats, '_psutil', return_value=ps), \
         patch.object(subprocess, 'run', side_effect=OSError('no nvidia-smi')):
        system_stats.machine_stats()
        system_stats.reset_cache()
        data = system_stats.machine_stats()
    assert data['cpu_percent'] == 0


def test_psutil_missing_leaves_the_gpu_half_standing():
    """A slimmed-down venv costs two numbers, not a 500."""
    with patch.object(system_stats, '_psutil', return_value=None), \
         patch.object(subprocess, 'run', return_value=_Proc('12, 2048, 24564\n')):
        data = system_stats.machine_stats()
    assert 'cpu_percent' not in data and 'ram_total_gb' not in data
    assert data['gpu_percent'] == 12


def test_the_route_always_answers_200_even_when_nothing_can_be_measured(app, client):
    """A glance, never a gate: a machine that can answer nothing answers {}."""
    with patch.object(system_stats, '_psutil', return_value=None), \
         patch.object(subprocess, 'run', side_effect=OSError('no nvidia-smi')):
        res = client.get('/api/system/stats')
    assert res.status_code == 200
    assert res.get_json() == {}


def test_the_route_serves_the_service_payload(app, client):
    with patch.object(subprocess, 'run', return_value=_Proc('55, 8192, 24564\n')):
        res = client.get('/api/system/stats')
    body = res.get_json()
    assert res.status_code == 200
    assert body['gpu_percent'] == 55
    assert body['vram_used_gb'] == 8.0
    assert body['vram_total_gb'] == 24.0
