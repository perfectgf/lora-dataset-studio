"""Exact, externally pinned exceptions for reviewed public plugin ports.

The digest and review reference are trust inputs supplied by the operator. This
module verifies their scope, not the human review itself; it never grants trust
to a manifest discovered in the candidate checkout.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import struct
import zipfile
import zlib
from dataclasses import dataclass
from email.parser import Parser
from email.policy import default as email_policy
from pathlib import Path

if __package__:
    from .private_plugin_policy import private_plugin_path_reason
else:
    from private_plugin_policy import private_plugin_path_reason

OID = re.compile(r'(?:[0-9a-f]{40}|[0-9a-f]{64})\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
PUBLIC_PRODUCTS = frozenset({
    'api_engines', 'camera_angles', 'canvas', 'civitai_publish', 'cloud_training',
    'dlss5', 'hf_publish', 'image_upscale', 'live', 'model_tools', 'resource_monitor',
    'qwen_dataset', 'remove_background', 'scrape', 'seedvr2', 'video',
})
ARCHIVE_SUFFIXES = ('.ldsplugin', '.zip', '.whl', '.tar', '.gz', '.bz2', '.xz',
                    '.tgz', '.tbz2', '.txz', '.7z', '.rar', '.pyz')
WHEEL_NAME = re.compile(r'(?P<distribution>[A-Za-z0-9_]+)-(?P<version>[A-Za-z0-9_.+]+)'
                        r'(?:-[0-9][A-Za-z0-9_]*)?-py3-none-any\.whl\Z')
WHEEL_NATIVE_SUFFIXES = ('.exe', '.dll', '.pyd', '.dylib', '.so', '.pyc', '.pyo', '.bat', '.cmd', '.ps1', '.pth')
WHEEL_MAX_BYTES = 8 * 1024 * 1024
WHEEL_MAX_EXPANDED = 32 * 1024 * 1024
WHEEL_MAX_MEMBER = 4 * 1024 * 1024
WHEEL_MAX_MEMBERS = 512
WINDOWS_RESERVED = re.compile(r'(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|\Z)', re.I)


class ManifestError(ValueError):
    pass


def require(condition, message='Invalid approved public-port manifest.'):
    if not condition:
        raise ManifestError(message)


def strict_json(content: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON field in publication policy.')
            result[key] = value
        return result

    return json.loads(content.decode('utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'Non-finite JSON value.'))


def fields(value, names):
    require(type(value) is dict and set(value) == set(names))


def oid(value):
    require(type(value) is str and OID.fullmatch(value))
    return value


def digest(value):
    require(type(value) is str and SHA256.fullmatch(value))
    return value


def git_path(value):
    require(type(value) is str and value and '\\' not in value)
    require(all(part not in ('', '.', '..') for part in value.split('/')))
    require(all(32 <= ord(char) < 0xD800 or 0xDFFF < ord(char) <= 0x10FFFF
                for char in value))
    require('\x7f' not in value and ':' not in value)
    return value


def wheel_member_path(value):
    """Use portable, unambiguous file names; this policy never extracts a wheel."""
    git_path(value)
    require(len(value) <= 240)
    for part in value.split('/'):
        require(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.+\-]{0,79}', part)
                and not part.endswith('.') and not WINDOWS_RESERVED.match(part))
    require(not value.casefold().endswith(ARCHIVE_SUFFIXES),
            'Nested archives cannot receive generated-resource exceptions.')
    require(private_plugin_path_reason(value) is None,
            'Private product paths cannot be embedded in a generated wheel.')
    require(not value.casefold().endswith(WHEEL_NATIVE_SUFFIXES)
            and not re.search(r'\.so(?:\.[0-9]+)+\Z', value.casefold())
            and not value.split('/')[0].casefold().endswith('.data'),
            'Native binaries and installation hooks are outside the pure-Python resource exception.')
    return value


def wheel_descriptor(entry):
    """The only archive exception: an explicit generated, pure-Python wheel."""
    parts = entry['path'].split('/')
    require(len(parts) == 5 and parts[2:4] == ['resources', 'wheels']
            and WHEEL_NAME.fullmatch(parts[4]) and entry['mode'] == '100644',
            'Generated wheel exceptions require an exact resources/wheels path.')
    provenance = entry['provenance']
    require(provenance['kind'] == 'generated')
    resource = provenance['resource']
    fields(resource, ('type', 'sha256', 'size', 'members', 'external_source'))
    require(resource['type'] == 'wheel')
    # This is an immutable upstream reference for the human review, not a claim
    # that its bytes were present in main or downloaded/verified by this guard.
    source = resource['external_source']
    fields(source, ('type', 'url', 'sha256'))
    require(source['type'] == 'pypi_sdist' and type(source['url']) is str
            and re.fullmatch(r'https://files\.pythonhosted\.org/packages/'
                             r'[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{60}/'
                             r'[A-Za-z0-9_][A-Za-z0-9_.+\-]*\.tar\.gz', source['url']))
    digest(source['sha256'])
    digest(resource['sha256'])
    require(type(resource['size']) is int and 22 <= resource['size'] <= WHEEL_MAX_BYTES)
    members = resource['members']
    require(type(members) is list and 0 < len(members) <= WHEEL_MAX_MEMBERS)
    seen, expanded = set(), 0
    for member in members:
        fields(member, ('path', 'size', 'sha256'))
        name = wheel_member_path(member['path'])
        require(name.casefold() not in seen, 'Duplicate generated-resource member.')
        seen.add(name.casefold())
        require(type(member['size']) is int and 0 <= member['size'] <= WHEEL_MAX_MEMBER)
        expanded += member['size']
        digest(member['sha256'])
    require(expanded <= WHEEL_MAX_EXPANDED)
    require(all('/'.join(name.split('/')[:index]) not in seen
                for name in seen for index in range(1, len(name.split('/')))),
            'A wheel file cannot also be a directory or its case alias.')


def verify_wheel_metadata(members, filename):
    """Identify the narrow pure-Python wheel format, without installing it."""
    name = WHEEL_NAME.fullmatch(filename)
    require(name is not None)
    distribution, version = name.group('distribution', 'version')
    directory = distribution + '-' + version + '.dist-info'
    metadata_path, wheel_path = directory + '/METADATA', directory + '/WHEEL'
    require({metadata_path, wheel_path, directory + '/RECORD'} <= set(members),
            'Generated wheel metadata is missing or inconsistent with its filename.')
    require({path.split('/')[0] for path in members if path.split('/')[0].endswith('.dist-info')} == {directory},
            'A generated wheel must describe exactly one distribution.')
    parsed = []
    for path in (metadata_path, wheel_path):
        message = Parser(policy=email_policy).parsestr(members[path].decode('utf-8'))
        require(not message.defects, 'Malformed generated wheel metadata.')
        parsed.append(message)
    metadata, wheel = parsed

    def single(message, field):
        values = message.get_all(field, [])
        require(len(values) == 1, 'Missing or duplicate generated wheel metadata field.')
        return str(values[0]).strip()

    project = single(metadata, 'Name')
    require(re.fullmatch(r'[0-9]+\.[0-9]+', single(metadata, 'Metadata-Version')))
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', project))
    require(re.sub(r'[-_.]+', '-', project).casefold() == re.sub(r'[-_.]+', '-', distribution).casefold()
            and single(metadata, 'Version') == version,
            'Generated wheel distribution metadata differs from its filename.')
    require(single(wheel, 'Wheel-Version') == '1.0'
            and single(wheel, 'Root-Is-Purelib').casefold() == 'true'
            and single(wheel, 'Tag') == 'py3-none-any'
            and not wheel.get_payload().strip(),
            'Generated resource must declare a pure-Python wheel without an executable payload.')


def verify_wheel(content, resource, filename):
    """Verify every Git-blob byte and every member, without imports or extraction.

    Only simple ZIP32 STORED/DEFLATED files are accepted. Reject ZIP metadata or
    unused compressed bytes that could hide material outside the reviewed files.
    """
    require(len(content) == resource['size']
            and hashlib.sha256(content).hexdigest() == resource['sha256'],
            'The generated wheel differs from its approved bytes.')
    try:
        end = struct.unpack('<4s4H2IH', content[-22:])
        signature, disk, directory_disk, disk_count, count, directory_size, start, comment = end
        require(signature == b'PK\x05\x06' and disk == directory_disk == comment == 0
                and disk_count == count == len(resource['members'])
                and start + directory_size == len(content) - 22)
        with zipfile.ZipFile(io.BytesIO(content)) as wheel:
            require(not wheel.comment and wheel.start_dir == start)
            infos = wheel.infolist()
            expected = {member['path']: member for member in resource['members']}
            require(len(infos) == count and {info.filename for info in infos} == set(expected),
                    'The generated wheel member inventory differs from its approval.')
            local_position, central_position = 0, start
            metadata_members = {}
            for info in infos:
                member = expected[info.filename]
                name = info.filename.encode('ascii')
                require(info.orig_filename == info.filename and not info.is_dir()
                        and not info.extra and not info.comment
                        and info.flag_bits in (0, 0x800)
                        and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and info.extract_version in (10, 20) and info.create_system in (0, 3)
                        and info.volume == 0 and info.internal_attr in (0, 1))
                mode = info.external_attr >> 16
                require(stat.S_IFMT(mode) in (0, stat.S_IFREG) and not mode & 0o7000
                        and not (info.external_attr & 0xffff) & ~0x20,
                        'Generated wheel members must be regular files, without reparse attributes.')
                require(info.file_size == member['size'] and info.header_offset == local_position)
                header = struct.unpack_from('<4s5H3I2H', content, local_position)
                require(header[0] == b'PK\x03\x04' and header[1] == info.extract_version
                        and header[2] == info.flag_bits and header[3] == info.compress_type
                        and header[6:9] == (info.CRC, info.compress_size, info.file_size)
                        and header[9:] == (len(name), 0))
                payload_start = local_position + 30 + len(name)
                payload_end = payload_start + info.compress_size
                require(content[local_position + 30:payload_start] == name and payload_end <= start)
                packed = content[payload_start:payload_end]
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    stream = zlib.decompressobj(-zlib.MAX_WBITS)
                    unpacked = stream.decompress(packed, member['size'] + 1)
                    require(stream.eof and not stream.unused_data and not stream.unconsumed_tail)
                else:
                    unpacked = packed
                require(len(unpacked) == member['size'] and zlib.crc32(unpacked) == info.CRC
                        and hashlib.sha256(unpacked).hexdigest() == member['sha256'],
                        'A generated wheel member differs from its approved content.')
                metadata_members[info.filename] = unpacked if info.filename.endswith(('/WHEEL', '/METADATA')) else b''
                # ZIP comments/extras and unlisted local records are forbidden.
                require(content[central_position:central_position + 4] == b'PK\x01\x02')
                lengths = struct.unpack_from('<3H', content, central_position + 28)
                require(lengths == (len(name), 0, 0)
                        and content[central_position + 46:central_position + 46 + len(name)] == name)
                central_position += 46 + len(name)
                local_position = payload_end
            require(local_position == start and central_position == len(content) - 22,
                    'Unreviewed data surrounds the generated wheel members.')
            verify_wheel_metadata(metadata_members, filename)
    except (zipfile.BadZipFile, struct.error, zlib.error, UnicodeError, NotImplementedError) as exc:
        raise ManifestError('Unsupported or malformed generated wheel.') from exc


def outside_checkout(repo, path, git):
    """Resolve aliases and inspect every worktree, while allowing shared Git storage."""
    require(path.is_absolute(), 'The approved manifest path must be absolute.')
    resolved = path.resolve(strict=True)
    common = Path(git(repo, 'rev-parse', '--path-format=absolute', '--git-common-dir').decode().strip()).resolve()
    require(resolved.is_file())
    if resolved.is_relative_to(common):
        return resolved
    for item in git(repo, 'worktree', 'list', '--porcelain', '-z').split(b'\0'):
        if item.startswith(b'worktree '):
            checkout = Path(item[len(b'worktree '):].decode('utf-8')).resolve()
            require(not resolved.is_relative_to(checkout),
                    'The approved manifest must be stored outside every checkout.')
    return resolved


@dataclass(frozen=True)
class ApprovedPort:
    content: bytes
    data: dict
    sha256: str
    review_sha256: str

    @property
    def tip(self):
        return self.data['tip']

    def exceptions(self, repo, tip, public_base, private_source, git, entries):
        """Validate the whole graph before returning exact contextual exceptions."""
        data = self.data
        require((tip, public_base, private_source) ==
                (data['tip'], data['public_base'], data['private_source']),
                'The approved manifest does not match the tip or installed trust inputs.')
        origin = data['public_origin']
        for commit in (tip, public_base, private_source, origin):
            require(git(repo, 'cat-file', '-t', commit).strip() == b'commit')
        # Failure is fatal; the original baseline remains the history scan boundary.
        git(repo, 'merge-base', '--is-ancestor', public_base, origin)
        git(repo, 'merge-base', '--is-ancestor', origin, tip)
        actual = set(git(repo, 'rev-list', tip, '--not', origin, '--').decode().splitlines())
        declared = {item['commit'] for item in data['commits']}
        require(actual == declared, 'The approved manifest does not match the complete export graph.')
        require(git(repo, 'rev-parse', tip + '^{tree}').decode().strip() == data['tip_tree'])
        public_entries = {name: (mode, kind, blob) for mode, kind, blob, name in entries(repo, origin)}
        exceptions = set()
        for item in data['commits']:
            commit = item['commit']
            require(git(repo, 'rev-parse', commit + '^{tree}').decode().strip() == item['tree'])
            parents = git(repo, 'show', '-s', '--format=%P', commit).decode().strip().split()
            require(parents == item['parents'], 'The approved commit parent graph differs.')
            actual_entries = {name: (mode, kind, blob) for mode, kind, blob, name in entries(repo, commit)}
            for exception in item['exceptions']:
                name, mode, blob = exception['path'], exception['mode'], exception['blob']
                require(actual_entries.get(name) == (mode, 'blob', blob),
                        'An approved entry does not match its exact commit/tree/path/mode/blob.')
                provenance = exception['provenance']
                for source in provenance['sources']:
                    require(public_entries.get(source['path']) == (source['mode'], 'blob', source['blob']),
                            'The declared public source does not exist in the public origin.')
                if provenance['kind'] == 'unchanged':
                    source = provenance['sources'][0]
                    require((mode, blob) == (source['mode'], source['blob']),
                            'An unchanged public port differs from its public source.')
                if 'resource' in provenance:
                    resource = provenance['resource']
                    # Bound the object before asking Git to materialize its bytes.
                    require(int(git(repo, 'cat-file', '-s', blob)) == resource['size'])
                    verify_wheel(git(repo, 'cat-file', 'blob', blob), resource, name.rsplit('/', 1)[-1])
                exceptions.add((commit, item['tree'], name, mode, 'blob', blob))
        return exceptions


def load_manifest(repo, path, expected_sha256, review_sha256, git):
    require(all(value is not None for value in (path, expected_sha256, review_sha256)),
            'Manifest path, pinned SHA256 and immutable review SHA256 are all required.')
    digest(expected_sha256)
    digest(review_sha256)
    resolved = outside_checkout(repo, Path(path), git)
    content = resolved.read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected_sha256,
            'The approved manifest bytes differ from their installed SHA256.')
    data = strict_json(content)
    fields(data, ('schema_version', 'public_base', 'private_source', 'public_origin',
                  'tip', 'tip_tree', 'review_sha256', 'products', 'commits'))
    require(type(data['schema_version']) is int and data['schema_version'] == 1)
    require(digest(data['review_sha256']) == review_sha256)
    for name in ('public_base', 'private_source', 'public_origin', 'tip', 'tip_tree'):
        oid(data[name])
    products = data['products']
    require(type(products) is list and products and all(type(p) is str for p in products))
    require(len(products) == len(set(products)) and set(products) <= PUBLIC_PRODUCTS)
    commits = data['commits']
    require(type(commits) is list and commits)
    seen_commits, seen_products = set(), set()
    for item in commits:
        fields(item, ('commit', 'tree', 'parents', 'exceptions'))
        commit = oid(item['commit'])
        require(commit not in seen_commits, 'Duplicate approved commit.')
        seen_commits.add(commit)
        oid(item['tree'])
        require(type(item['parents']) is list)
        parents = [oid(parent) for parent in item['parents']]
        require(len(parents) == len(set(parents)))
        require(type(item['exceptions']) is list)
        seen_paths = set()
        for entry in item['exceptions']:
            fields(entry, ('path', 'mode', 'type', 'blob', 'provenance'))
            name = git_path(entry['path'])
            parts = name.split('/')
            require(len(parts) >= 3 and parts[0] == 'bundled' and parts[1] in products)
            require(all(part.casefold() != 'bundled' for part in parts[2:]))
            require(name not in seen_paths, 'Duplicate approved path in a commit.')
            seen_paths.add(name)
            seen_products.add(parts[1])
            require(entry['mode'] in ('100644', '100755') and entry['type'] == 'blob',
                    'Only regular source files can receive public-port exceptions.')
            oid(entry['blob'])
            provenance = entry['provenance']
            require(type(provenance) is dict)
            if 'resource' in provenance:
                fields(provenance, ('kind', 'sources', 'evidence_sha256', 'resource'))
                wheel_descriptor(entry)
            else:
                fields(provenance, ('kind', 'sources', 'evidence_sha256'))
                require(not parts[-1].casefold().endswith(ARCHIVE_SUFFIXES),
                        'Distribution archives cannot receive source-port exceptions.')
            require(provenance['kind'] in ('unchanged', 'reviewed_port', 'generated',
                                           'reviewed_infrastructure', 'reviewed_product_release'))
            if provenance['kind'] == 'reviewed_product_release':
                # Explicit source publication, still pinned to exact reviewed
                # files and history. It grants no permission to other products.
                require(parts[1] == 'dlss5' and 'resource' not in provenance)
            digest(provenance['evidence_sha256'])
            sources = provenance['sources']
            require(type(sources) is list)
            if provenance['kind'] in ('reviewed_infrastructure', 'reviewed_product_release') or 'resource' in provenance:
                # Explicitly reviewed new infrastructure / externally sourced
                # generated assets must not fabricate a source blob from main.
                require(not sources)
            else:
                require(sources)
            require(provenance['kind'] != 'unchanged' or len(sources) == 1)
            seen_sources = set()
            for source in sources:
                fields(source, ('path', 'mode', 'blob'))
                source_path = git_path(source['path'])
                require('bundled' not in [part.casefold() for part in source_path.split('/')])
                require(not source_path.casefold().endswith('.ldsplugin')
                        and source_path.split('/')[-1].casefold() != 'transition-pack.zip')
                require(source_path not in seen_sources)
                seen_sources.add(source_path)
                require(source['mode'] in ('100644', '100755'))
                oid(source['blob'])
    require(seen_products == set(products), 'The product inventory must match the exact exceptions.')
    return ApprovedPort(content, data, expected_sha256, review_sha256)
