import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const venv = join(root, '.venv');
const venvPython = process.platform === 'win32'
  ? join(venv, 'Scripts', 'python.exe')
  : join(venv, 'bin', 'python');

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, {
    cwd: root,
    stdio: 'inherit',
    shell: false,
    ...opts,
  });
  if (res.status !== 0) {
    process.exit(res.status ?? 1);
  }
}

function runCapture(cmd, args) {
  return spawnSync(cmd, args, {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
    shell: false,
  });
}

function parseVersion(text) {
  const m = String(text || '').match(/(\d+)\.(\d+)\.(\d+)/);
  if (!m) return null;
  return { major: Number(m[1]), minor: Number(m[2]), patch: Number(m[3]) };
}

function supportedPython(version) {
  return version && version.major === 3 && version.minor >= 11 && version.minor <= 14;
}

function pythonCandidates() {
  const candidates = [];
  if (process.env.PYTHON) candidates.push([process.env.PYTHON, []]);
  if (process.platform === 'win32') {
    candidates.push(['py', ['-3.14']], ['py', ['-3.13']], ['py', ['-3.12']], ['py', ['-3.11']]);
  }
  candidates.push(['python3.14', []], ['python3.13', []], ['python3.12', []], ['python3.11', []], ['python3', []], ['python', []]);
  return candidates;
}

function findPython() {
  for (const [cmd, prefixArgs] of pythonCandidates()) {
    const res = runCapture(cmd, [...prefixArgs, '-c', 'import sys; print(".".join(map(str, sys.version_info[:3])))']);
    if (res.status !== 0) continue;
    const version = parseVersion(res.stdout);
    if (supportedPython(version)) return { cmd, prefixArgs, version };
  }
  console.error('Infinite Memory requires Python 3.11, 3.12, 3.13, or 3.14. Install one of those versions and rerun npm install.');
  process.exit(1);
}

if (!existsSync(venvPython)) {
  const py = findPython();
  console.log(`Creating Infinite Memory Python venv with ${py.cmd} ${py.prefixArgs.join(' ')} (${py.version.major}.${py.version.minor}.${py.version.patch})`);
  run(py.cmd, [...py.prefixArgs, '-m', 'venv', venv]);
}

run(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip']);
run(venvPython, ['-m', 'pip', 'install', '-e', '.']);
