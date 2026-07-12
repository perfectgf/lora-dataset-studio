"""Thin vast.ai REST client (no SDK dependency). All vast-specific HTTP lives
here so an API change touches one file. The API key is read from the secret
store on every call — never cached, so a key pasted in Settings applies
immediately."""
import logging

import requests

from .. import config as cfg

logger = logging.getLogger(__name__)

API_BASE = 'https://console.vast.ai/api/v0'
_TIMEOUT = 30


class VastError(RuntimeError):
    pass


def _request(method, path, **kwargs):
    key = cfg.secret('VAST_API_KEY')
    if not key:
        raise VastError('VAST_API_KEY is not configured')
    headers = {'Authorization': f'Bearer {key}', 'Accept': 'application/json'}
    return requests.request(method, f'{API_BASE}{path}', headers=headers,
                            timeout=_TIMEOUT, **kwargs)


def search_offers(min_vram_gb: int, max_dph: float, limit: int = 20) -> list:
    """Verified-datacenter offers with enough VRAM under the price cap,
    cheapest first. gpu_ram is expressed in MB on the vast side."""
    body = {
        'gpu_ram': {'gte': int(min_vram_gb) * 1024},
        'reliability': {'gte': 0.95},
        'verified': {'eq': True},
        'rentable': {'eq': True},
        'dph_total': {'lte': float(max_dph)},
        'num_gpus': {'eq': 1},
        'type': 'ondemand',
        'limit': int(limit),
    }
    r = _request('POST', '/bundles/', json=body)
    if r.status_code != 200:
        raise VastError(f'offer search failed: HTTP {r.status_code}')
    offers = (r.json() or {}).get('offers') or []
    out = [{
        'offer_id': o.get('id'),
        'gpu_name': o.get('gpu_name'),
        'dph_total': o.get('dph_total'),
        'gpu_ram_gb': round((o.get('gpu_ram') or 0) / 1024.0, 1),
    } for o in offers if o.get('id') is not None]
    out.sort(key=lambda x: x['dph_total'] if x['dph_total'] is not None else 9e9)
    return out


def create_instance(offer_id, image: str, env: dict, disk_gb: int, label: str,
                    onstart: str | None = None) -> str:
    body = {'image': image, 'label': label, 'disk': int(disk_gb),
            'runtype': 'args', 'env': dict(env or {})}
    if onstart:
        body['onstart'] = onstart
    r = _request('PUT', f'/asks/{offer_id}/', json=body)
    data = r.json() if r.status_code == 200 else {}
    if r.status_code != 200 or not data.get('success'):
        raise VastError(f'create_instance failed: HTTP {r.status_code} {data}')
    return str(data.get('new_contract'))


def list_instances() -> list:
    r = _request('GET', '/instances/')
    if r.status_code != 200:
        raise VastError(f'list_instances failed: HTTP {r.status_code}')
    out = []
    for i in (r.json() or {}).get('instances') or []:
        out.append({
            'instance_id': str(i.get('id')),
            'actual_status': i.get('actual_status'),
            'public_ipaddr': i.get('public_ipaddr'),
            'ports': i.get('ports'),
            'label': i.get('label'),
            'dph_total': i.get('dph_total'),
        })
    return out


def get_instance(instance_id):
    for inst in list_instances():
        if inst['instance_id'] == str(instance_id):
            return inst
    return None


def destroy_instance(instance_id) -> bool:
    """Idempotent: a 404 means the instance is already gone — success."""
    try:
        r = _request('DELETE', f'/instances/{instance_id}/')
    except VastError:
        raise
    except Exception as e:                      # network blip: caller may retry
        logger.warning('destroy_instance %s: %s', instance_id, e)
        return False
    if r.status_code in (200, 404):
        return True
    logger.warning('destroy_instance %s: HTTP %s', instance_id, r.status_code)
    return False


def derive_base_url(instance: dict, container_port: int):
    """Public URL of the pod's UI from the docker-style port mapping.
    Returns None while the mapping isn't published yet (instance booting)."""
    if not instance:
        return None
    ip = instance.get('public_ipaddr')
    ports = instance.get('ports') or {}
    entries = ports.get(f'{container_port}/tcp') or []
    if not ip or not entries:
        return None
    host_port = (entries[0] or {}).get('HostPort')
    return f'http://{ip}:{host_port}' if host_port else None
