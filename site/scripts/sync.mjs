// Prebuild: pull everything the site shows from the repo's sources of truth, so nothing is copied by hand.
//
//   docs/ledger.md              -> src/generated/ledger.json      (master table: status, audit note, figures)
//   docs/programmes/*.md        -> src/generated/programmes.json  (title, status line, experiments, theory threads)
//   celestrium/watchlist.json   -> src/generated/calendar.json + public/calendar.ics
//   git log -- docs/            -> src/generated/changes.json     (what changed, for the hub and /feed.xml)
//   docs/figures/**             -> public/fig/**  (web size, <= 1400 px), public/fig/_full/** (originals),
//                                  public/fig/_t/**.webp (thumbnails), src/generated/figures.json (provenance)
//
// Only figures the site references are copied: tokens like name.png in src/ and the ledger's figure column.
// Runs automatically before `astro build` (npm "prebuild"). Needs only node and git; sharp is optional.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = path.resolve(SITE, '..');
const FIGS = path.join(ROOT, 'docs', 'figures');
const GEN = path.join(SITE, 'src', 'generated');
const PUB = path.join(SITE, 'public');
const REPO = 'https://github.com/sub-surface/astro-theory';
const WEB_WIDTH = 1400;
const THUMB_WIDTH = 260;

fs.mkdirSync(GEN, { recursive: true });
const read = (p) => fs.readFileSync(p, 'utf8');
const write = (name, obj) => fs.writeFileSync(path.join(GEN, name), JSON.stringify(obj, null, 1));
const git = (...args) => {
  try {
    return execFileSync('git', args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'ignore'], timeout: 60000 }).toString();
  } catch {
    return '';
  }
};

// ------------------------------------------------------------------ markdown cell -> plain text
const TEX = [
  [/\\text\{([^}]*)\}/g, '$1'], [/\\mathbf\{([^}]*)\}/g, '$1'], [/\\mathrm\{([^}]*)\}/g, '$1'],
  [/\\scriptscriptstyle\\text\{([^}]*)\}/g, '$1'], [/\\scriptscriptstyle/g, ''], [/\\rm\s*/g, ''],
  [/\^\\circ/g, '°'], [/\\circ/g, '°'], [/\\in\b/g, '∈'], [/\\le(q)?\b/g, '≤'], [/\\ge(q)?\b/g, '≥'],
  [/\\times/g, '×'], [/\\sigma/g, 'σ'], [/\\Lambda/g, 'Λ'], [/\\lambda/g, 'λ'], [/\\alpha/g, 'α'], [/\\chi/g, 'χ'],
  [/\\Delta/g, 'Δ'], [/\\hat\{([^}]*)\}/g, '$1'], [/\\bar\{([^}]*)\}/g, '$1'], [/\\pm/g, '±'], [/\\min/g, 'min'], [/\\max/g, 'max'], [/\\approx/g, '≈'],
  [/\\,/g, ' '], [/\\;/g, ' '], [/\\ /g, ' '], [/\\%/g, '%'], [/\\mu/g, 'μ'], [/\\ell/g, 'ℓ'],
  [/_\{([^}]*)\}/g, '_$1'], [/\^\{([^}]*)\}/g, '^$1'], [/\{|\}/g, ''],
];
const plain = (s) => {
  let t = s.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/\*\*/g, '').replace(/`/g, '').replace(/\*/g, '');
  t = t.replace(/\$([^$]*)\$/g, (_, m) => TEX.reduce((acc, [re, rep]) => acc.replace(re, rep), m));
  return t.replace(/\s+/g, ' ').trim();
};

// ------------------------------------------------------------------ ledger
const STATUS_CLASS = [
  [/WITHDRAWN|NOT DEMONSTRATED/, 'no'], [/SYNTHETIC ONLY/, 'syn'], [/NEEDS RERUN/, 'rerun'],
  [/PLANNED/, 'plan'], [/SOUND/, 'ok'],
];
const ledgerRows = [];
{
  const lines = read(path.join(ROOT, 'docs', 'ledger.md')).split('\n');
  const head = lines.findIndex((l) => l.startsWith('| ID | Title & Focus |'));
  const cols = lines[head].split('|').slice(1, -1).map((c) => c.trim());
  const at = (name) => cols.findIndex((c) => c.startsWith(name));
  const I = { id: at('ID'), title: at('Title'), data: at('Primary Data'), code: at('Checkpoint'), status: at('Status'),
              note: at('Audit note'), metric: at('Key Metric'), fig: at('Diagnostic Figure') };
  for (const l of lines.slice(head + 2)) {
    if (!l.startsWith('| **EXP-')) break;
    const c = l.split(' | ').map((x) => x.replace(/^\|\s*/, '').replace(/\s*\|$/, ''));
    const statusBold = (c[I.status].match(/\*\*([^*]+)\*\*/) || [, c[I.status]])[1].trim();
    const links = [...c[I.fig].matchAll(/\]\(\.\/([^)]+)\)/g)].map((m) => m[1]);
    ledgerRows.push({
      id: plain(c[I.id]).replace('EXP-2026-', ''),
      title: plain(c[I.title]), data: plain(c[I.data]), status: statusBold,
      cls: (STATUS_CLASS.find(([re]) => re.test(statusBold)) || [, 'plan'])[1],
      note: plain(c[I.note]), metric: plain(c[I.metric]),
      figures: links.filter((p) => p.startsWith('figures/')).map((p) => p.slice('figures/'.length)),
      results: links.filter((p) => p.startsWith('results/')).map((p) => `${REPO}/blob/main/docs/${p}`),
    });
  }
}

// ------------------------------------------------------------------ programmes
const programmes = [];
for (const f of fs.readdirSync(path.join(ROOT, 'docs', 'programmes')).filter((x) => x.endsWith('.md')).sort()) {
  const md = read(path.join(ROOT, 'docs', 'programmes', f));
  const title = md.match(/^# (P\d) · (.+)$/m);
  if (!title) continue;
  const status = (md.match(/^\*\*Status\*\*:\s*(.+)$/m) || [, ''])[1];
  const ledgerLine = (md.match(/^\*\*Ledger\*\*:\s*(.+)$/m) || [, ''])[1];
  const ids = [];
  const m = ledgerLine.match(/EXP-2026-([A-Z0-9]+)((?:,\s*[A-Z0-9]{1,2}\b)*)/);
  if (m) {
    ids.push(m[1]);
    for (const x of m[2].split(',').map((s) => s.trim()).filter(Boolean)) ids.push(x);
  }
  const threads = [...md.matchAll(/^## (T\d+) · (.+)$/gm)].map((t) => ({ id: t[1], title: plain(t[2]) }));
  programmes.push({ id: title[1], title: plain(title[2]), file: `docs/programmes/${f}`, url: `${REPO}/blob/main/docs/programmes/${f}`,
                    status: plain(status.split(' · ')[0]).split(/[.;:]\s/)[0].replace(/[;.:]$/, ''), experiments: ids, threads });
}

// ------------------------------------------------------------------ calendar (+ .ics)
const cal = JSON.parse(read(path.join(ROOT, 'celestrium', 'watchlist.json'))).events;
{
  const esc = (s) => s.replace(/\\/g, '\\\\').replace(/;/g, '\\;').replace(/,/g, '\\,').replace(/\n/g, '\\n');
  const d8 = (iso) => iso.replaceAll('-', '');
  const next = (iso) => { const d = new Date(iso + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + 1); return d.toISOString().slice(0, 10); };
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+/, '');
  const ics = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Celestrium//watchlist//EN', 'X-WR-CALNAME:Celestrium observing calendar'];
  for (const e of cal.filter((x) => x.start)) {
    const desc = [e.check, e.test ? 'TEST: ' + e.test : '', e.source].filter(Boolean).join(' | ');
    ics.push('BEGIN:VEVENT', `UID:${e.id}@celestrium`, `DTSTAMP:${stamp}`, `DTSTART;VALUE=DATE:${d8(e.start)}`,
             `DTEND;VALUE=DATE:${d8(next(e.end || e.start))}`, `SUMMARY:${esc(`[${e.kind}] ${e.target}: ${e.title}`)}`,
             `DESCRIPTION:${esc(desc)}`, 'END:VEVENT');
  }
  ics.push('END:VCALENDAR');
  fs.writeFileSync(path.join(PUB, 'calendar.ics'), ics.join('\r\n') + '\r\n');
}

// ------------------------------------------------------------------ what changed (git history of docs/)
if (git('rev-parse', '--is-shallow-repository').trim() === 'true') git('fetch', '--unshallow', '--quiet');
const changes = git('log', '-40', '--date=short', '--format=%H%x1f%ad%x1f%s%x1f%b%x1e', '--', 'docs', 'celestrium/watchlist.json')
  .split('\x1e').map((r) => r.trim()).filter(Boolean).map((r) => {
    const [sha, date, subject, body] = r.split('\x1f');
    const para = (body || '').split(/\n\s*\n/)[0].replace(/\s+/g, ' ').trim();
    return { sha, date, subject, summary: /^Co-Authored-By|^Claude-Session/.test(para) ? '' : para.slice(0, 400),
             url: `${REPO}/commit/${sha}` };
  });

// ------------------------------------------------------------------ figures
const allFigs = new Map(); // basename -> [relpaths]
const walk = (dir) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (/\.(png|gif|jpe?g|mp4)$/i.test(e.name)) {
      const rel = path.relative(FIGS, p).split(path.sep).join('/');
      allFigs.set(e.name, [...(allFigs.get(e.name) || []), rel]);
    }
  }
};
walk(FIGS);
const wanted = new Set();
const resolve = (tok) => {
  const t = tok.replace(/^\/?fig\//, '');
  if (fs.existsSync(path.join(FIGS, t))) return t;
  const hits = allFigs.get(path.basename(t)) || [];
  return hits.length === 1 ? hits[0] : null;
};
const srcFiles = [];
const walkSrc = (dir) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory() && e.name !== 'generated') walkSrc(p);
    else if (/\.(astro|ts|mjs|json)$/.test(e.name)) srcFiles.push(p);
  }
};
walkSrc(path.join(SITE, 'src'));
const unresolved = [];
for (const f of srcFiles) {
  for (const m of read(f).matchAll(/[\w./-]+\.(?:png|gif|jpe?g)\b/g)) {
    if (m[0].includes('favicon') || m[0].includes('apple-touch')) continue;
    const r = resolve(m[0]);
    if (r) wanted.add(r);
    else if (/\/fig\/|^[\w-]+\.(png|gif)$/.test(m[0])) unresolved.push(`${path.relative(SITE, f)}: ${m[0]}`);
  }
}
for (const row of ledgerRows) for (const f of row.figures) { const r = resolve(f); if (r) wanted.add(r); }
for (const r of [...wanted]) {           // a GIF ships with its MP4 sibling when one exists
  if (r.endsWith('.gif') && fs.existsSync(path.join(FIGS, r.replace(/\.gif$/, '.mp4')))) wanted.add(r.replace(/\.gif$/, '.mp4'));
}

let sharp = null;
try { sharp = (await import('sharp')).default; } catch { console.warn('[sync] sharp unavailable: figures copied at full size, no thumbnails'); }

// provenance: which scripts write each figure (by file name)
const codeFiles = [];
const walkCode = (dir) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory() && !['__pycache__', 'node_modules'].includes(e.name)) walkCode(p);
    else if (e.name.endsWith('.py')) codeFiles.push([path.relative(ROOT, p).split(path.sep).join('/'), read(p)]);
  }
};
walkCode(path.join(ROOT, 'scripts'));
walkCode(path.join(ROOT, 'celestrium'));

const figures = {};
const outPath = (...p) => { const o = path.join(PUB, 'fig', ...p); fs.mkdirSync(path.dirname(o), { recursive: true }); return o; };
const fresh = (src, dst) => fs.existsSync(dst) && fs.statSync(dst).mtimeMs >= fs.statSync(src).mtimeMs;
for (const rel of [...wanted].sort()) {
  const src = path.join(FIGS, rel);
  const base = path.basename(rel);
  const stem = base.replace(/\.\w+$/, '');
  const info = { bytes: fs.statSync(src).size,
                 scripts: codeFiles.filter(([, txt]) => txt.includes(base) || txt.includes(`"${stem}"`)).map(([p]) => p) };
  const web = outPath(rel);
  if (sharp && /\.png$/i.test(rel)) {
    const meta = await sharp(src).metadata();
    info.width = meta.width; info.height = meta.height;
    if (!fresh(src, web)) {
      await sharp(src).resize({ width: WEB_WIDTH, withoutEnlargement: true }).png({ compressionLevel: 9, palette: true, quality: 95 }).toFile(web);
    }
    const thumb = outPath('_t', rel.replace(/\.png$/i, '.webp'));
    if (!fresh(src, thumb)) await sharp(src).resize({ width: THUMB_WIDTH }).webp({ quality: 80 }).toFile(thumb);
    info.thumb = `/fig/_t/${rel.replace(/\.png$/i, '.webp')}`;
    if (meta.width > WEB_WIDTH) {
      const full = outPath('_full', rel);
      if (!fresh(src, full)) fs.copyFileSync(src, full);
      info.full = `/fig/_full/${rel}`;
    }
  } else if (!fresh(src, web)) {
    fs.copyFileSync(src, web);
  }
  info.web_bytes = fs.statSync(web).size;
  figures[rel] = info;
}

write('ledger.json', ledgerRows);
write('programmes.json', programmes);
write('calendar.json', cal);
write('changes.json', changes);
write('figures.json', figures);
const kb = Object.values(figures).reduce((a, f) => a + f.web_bytes, 0) / 1024;
console.log(`[sync] ledger ${ledgerRows.length} rows, ${programmes.length} programmes, ${cal.length} calendar events, ` +
            `${changes.length} changes, ${Object.keys(figures).length} figures (${kb.toFixed(0)} kB web)`);
if (unresolved.length) {
  console.error('[sync] figure references with no file in docs/figures:\n  ' + unresolved.join('\n  '));
  process.exit(1);
}
