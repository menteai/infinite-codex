#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const venvPython = process.platform === 'win32'
  ? join(root, '.venv', 'Scripts', 'python.exe')
  : join(root, '.venv', 'bin', 'python');

if (!existsSync(venvPython)) {
  console.error('Infinite Memory is not initialized. Run `npm install` again in the package directory.');
  process.exit(1);
}

const res = spawnSync(venvPython, ['-m', 'infinite_memory.cli', ...process.argv.slice(2)], {
  cwd: root,
  stdio: 'inherit',
  env: process.env,
});

process.exit(res.status ?? 1);
