// Build provenance for the footer: which commit the deployed site was built from.
// Evaluated once per build. GIT_SHA (or Cloudflare's WORKERS_CI_COMMIT_SHA) overrides the local git lookup,
// for builds that run without a .git directory.
import { execFileSync } from 'node:child_process';

const git = (...args: string[]): string => {
  try {
    return execFileSync('git', args, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return '';
  }
};

export const REPO = 'https://github.com/sub-surface/astro-theory';

const full = process.env.GIT_SHA || process.env.WORKERS_CI_COMMIT_SHA || git('rev-parse', 'HEAD');
const fromEnv = Boolean(process.env.GIT_SHA || process.env.WORKERS_CI_COMMIT_SHA);

export const sha = full.slice(0, 7);
// Uncommitted changes under site/ (the build's working directory) mean the deployed pages are not exactly `sha`.
export const dirty = !fromEnv && git('status', '--porcelain', '--', '.') !== '';
// A commit that is not yet on GitHub would 404, so link to the history instead.
export const pushed = fromEnv || (full !== '' && git('branch', '-r', '--contains', full) !== '');
export const commitUrl = !sha ? REPO : pushed ? `${REPO}/commit/${full}` : `${REPO}/commits/main`;
export const builtOn = new Date().toISOString().slice(0, 10);

if (dirty) console.warn(`[build-info] building ${sha} with uncommitted changes in site/: footer shows ${sha}+`);
if (sha && !pushed) console.warn(`[build-info] ${sha} is not on origin yet: footer links to the commit list`);
