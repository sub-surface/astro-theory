// Atom feed of what changed in the research record (git history of docs/ and the calendar), built by scripts/sync.mjs.
import CHANGES from '../generated/changes.json';

const SITE = 'https://astro.subsurfaces.net';
const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

export function GET() {
  const items = CHANGES.slice(0, 30);
  const updated = (items[0]?.date || new Date().toISOString().slice(0, 10)) + 'T00:00:00Z';
  const xml = `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Celestrium: what changed</title>
  <subtitle>Experiments, audits and dates from an astrophysics notebook kept by Leon and Claude</subtitle>
  <link href="${SITE}/" />
  <link rel="self" href="${SITE}/feed.xml" />
  <id>${SITE}/</id>
  <updated>${updated}</updated>
  <author><name>Leon and Claude</name></author>
${items.map((c) => `  <entry>
    <title>${esc(c.subject)}</title>
    <link href="${c.url}" />
    <id>${c.url}</id>
    <updated>${c.date}T00:00:00Z</updated>
    <summary>${esc(c.summary || c.subject)}</summary>
  </entry>`).join('\n')}
</feed>
`;
  return new Response(xml, { headers: { 'Content-Type': 'application/atom+xml; charset=utf-8' } });
}
