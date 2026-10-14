"""Ship: test, check the tree is committed, build the site, push. Cloudflare builds and deploys on push.

    python scripts/infra/ship.py            # tests + site build + git push  (Cloudflare deploys from GitHub)
    python scripts/infra/ship.py --deploy   # also deploy from here with wrangler (if the git-triggered build is off)
    python scripts/infra/ship.py --no-push  # everything except the push

Refuses to ship uncommitted changes: the site footer names the commit it was built from, so what is live
should be exactly a commit on GitHub. Stop any `astro preview` first (it locks site/node_modules).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    exe = shutil.which(cmd[0]) or cmd[0]  # npm/npx are .cmd shims on Windows
    r = subprocess.run([exe, *cmd[1:]], cwd=ROOT, **kw)
    if r.returncode:
        sys.exit(f"ship: `{' '.join(cmd)}` failed ({r.returncode})")
    return r


def out(*cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", action="store_true", help="also run `npx wrangler deploy` from here")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    dirty = out("git", "status", "--porcelain", "--untracked-files=no")
    if dirty:
        sys.exit("ship: uncommitted changes to tracked files; commit or stash first:\n" + dirty)
    branch = out("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        sys.exit(f"ship: on branch {branch}; Cloudflare deploys main")

    run(sys.executable, "-m", "pytest", "-q", "-m", "not slow")
    run("npm", "run", "build")  # the same command Cloudflare runs on a fresh clone
    if not a.no_push:
        run("git", "push", "origin", "main")
        print("pushed: Cloudflare Workers Builds deploys this commit (dashboard: Workers & Pages > astro > Deployments)")
    if a.deploy:
        run("npx", "wrangler", "deploy")
    print("live: https://astro.subsurfaces.net  (footer should read", out("git", "rev-parse", "--short", "HEAD") + ")")


if __name__ == "__main__":
    main()
