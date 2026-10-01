/**
 * Celestrium Universal Wikilinks Cross-Referencing Utility
 *
 * Parses [[Target]] and [[Target|Display Text]] syntax across the site,
 * linking research papers, experiments, astronomical catalogs, and physical methods.
 */

export interface WikilinkMatch {
  raw: string;
  target: string;
  label: string;
  url: string;
  category: 'paper' | 'experiment' | 'object' | 'view' | 'concept' | 'external';
}

const WIKILINK_MAP: Record<string, { url: string; category: WikilinkMatch['category']; defaultLabel?: string }> = {
  // Research Papers
  'paper a': { url: '/papers/paper-a-cosmic-dipole/', category: 'paper', defaultLabel: 'Paper A (Cosmic Dipole)' },
  'paper-a': { url: '/papers/paper-a-cosmic-dipole/', category: 'paper', defaultLabel: 'Paper A' },
  'paper b': { url: '/papers/paper-b-continuous-flow-astrojev/', category: 'paper', defaultLabel: 'Paper B (Continuous Flow)' },
  'paper-b': { url: '/papers/paper-b-continuous-flow-astrojev/', category: 'paper', defaultLabel: 'Paper B' },
  'paper c': { url: '/papers/paper-c-autonomous-followup-mdp/', category: 'paper', defaultLabel: 'Paper C (Autonomous ToO)' },
  'paper-c': { url: '/papers/paper-c-autonomous-followup-mdp/', category: 'paper', defaultLabel: 'Paper C' },
  'paper d': { url: '/papers/paper-d-euclid-dr1-forecast/', category: 'paper', defaultLabel: 'Paper D (Euclid DR1 Forecast)' },
  'paper-d': { url: '/papers/paper-d-euclid-dr1-forecast/', category: 'paper', defaultLabel: 'Paper D' },
  'paper e': { url: '/papers/paper-e-frb-triage/', category: 'paper', defaultLabel: 'Paper E (FRB Evidential Triage)' },
  'paper-e': { url: '/papers/paper-e-frb-triage/', category: 'paper', defaultLabel: 'Paper E' },
  'paper f': { url: '/papers/paper-f-multi-tracer-bulk-flow/', category: 'paper', defaultLabel: 'Paper F (Multi-Tracer Bulk Flow)' },
  'paper-f': { url: '/papers/paper-f-multi-tracer-bulk-flow/', category: 'paper', defaultLabel: 'Paper F' },

  // Key Navigation Hubs
  'matrix': { url: '/matrix/', category: 'view', defaultLabel: 'Research Matrix' },
  'notice board': { url: '/matrix/', category: 'view', defaultLabel: 'Research Notice Board' },
  'research matrix': { url: '/matrix/', category: 'view', defaultLabel: 'Research Matrix & Notice Board' },
  'catalog': { url: '/catalog/', category: 'view', defaultLabel: 'Remarkable Objects Catalog' },
  'remarkable objects': { url: '/catalog/', category: 'view', defaultLabel: 'Remarkable Objects' },
  'taxonomy': { url: '/taxonomy/', category: 'view', defaultLabel: 'Field Taxonomy & Gaps' },
  'gaps': { url: '/taxonomy/', category: 'view', defaultLabel: 'Taxonomy Gaps' },
  'about': { url: '/about/', category: 'view', defaultLabel: 'About Celestrium' },
  'papers': { url: '/papers/', category: 'view', defaultLabel: 'Research Publications' },

  // Key Experiments
  'exp-2026-x': { url: '/papers/paper-e-frb-triage/', category: 'experiment', defaultLabel: 'EXP-2026-X (FRB Triage)' },
  'exp-x': { url: '/papers/paper-e-frb-triage/', category: 'experiment', defaultLabel: 'EXP-2026-X' },
  'exp-2026-w': { url: '/papers/paper-f-multi-tracer-bulk-flow/', category: 'experiment', defaultLabel: 'EXP-2026-W (Multi-Tracer MCMC)' },
  'exp-w': { url: '/papers/paper-f-multi-tracer-bulk-flow/', category: 'experiment', defaultLabel: 'EXP-2026-W' },
  'exp-2026-a': { url: '/matrix/#transients', category: 'experiment', defaultLabel: 'EXP-2026-A (Transient Stream)' },
  'exp-2026-b': { url: '/papers/paper-a-cosmic-dipole/', category: 'experiment', defaultLabel: 'EXP-2026-B (Quaia Dipole)' },
  'exp-2026-d': { url: '/papers/paper-a-cosmic-dipole/', category: 'experiment', defaultLabel: 'EXP-2026-D (Conformal Risk Control)' },
  'exp-2026-r': { url: '/papers/paper-c-autonomous-followup-mdp/', category: 'experiment', defaultLabel: 'EXP-2026-R (Multi-Messenger Triage)' },
  'exp-2026-v': { url: '/papers/paper-b-continuous-flow-astrojev/', category: 'experiment', defaultLabel: 'EXP-2026-V (Flow Matching)' },
  'exp-2026-01': { url: '/papers/paper-d-euclid-dr1-forecast/', category: 'experiment', defaultLabel: 'EXP-2026-01 (Euclid DR1 Injection)' },

  // Catalog Sources
  'obj-apex-01': { url: '/catalog/#OBJ-APEX-01', category: 'object', defaultLabel: 'OBJ-APEX-01 (Quaia Apex Quasar)' },
  'obj-kilonova-02': { url: '/catalog/#OBJ-KILONOVA-02', category: 'object', defaultLabel: 'OBJ-KILONOVA-02 (AT2017gfo Analog)' },
  'obj-highz-03': { url: '/catalog/#OBJ-HIGHZ-03', category: 'object', defaultLabel: 'OBJ-HIGHZ-03 (z=4.606 Quasar)' },
  'obj-hotdog-04': { url: '/catalog/#OBJ-HOTDOG-04', category: 'object', defaultLabel: 'OBJ-HOTDOG-04 (Extreme IR Quasar)' },
  'obj-halo-05': { url: '/catalog/#OBJ-HALO-05', category: 'object', defaultLabel: 'OBJ-HALO-05 (Fast Halo Subdwarf)' },
  'obj-lens-06': { url: '/catalog/#OBJ-LENS-06', category: 'object', defaultLabel: 'OBJ-LENS-06 (Quad Lens Quasar)' },

  // Core Physical & Methodological Concepts
  'cosmic dipole': { url: '/papers/paper-a-cosmic-dipole/', category: 'concept', defaultLabel: 'Cosmic Dipole' },
  'conformal risk control': { url: '/papers/paper-a-cosmic-dipole/', category: 'concept', defaultLabel: 'Conformal Risk Control' },
  'disentangled rlcd': { url: '/papers/paper-b-continuous-flow-astrojev/', category: 'concept', defaultLabel: 'Disentangled RLCD' },
  'rewarding doubt': { url: '/papers/paper-b-continuous-flow-astrojev/', category: 'concept', defaultLabel: 'Rewarding Doubt' },
  'fast radio bursts': { url: '/papers/paper-e-frb-triage/', category: 'concept', defaultLabel: 'Fast Radio Bursts (FRBs)' },
  'frb': { url: '/papers/paper-e-frb-triage/', category: 'concept', defaultLabel: 'FRBs' },
  'macquart relation': { url: '/papers/paper-e-frb-triage/', category: 'concept', defaultLabel: 'Macquart Relation' },
  'active 3d tiling': { url: '/papers/paper-c-autonomous-followup-mdp/', category: 'concept', defaultLabel: 'Active 3D GW Tiling' },
  'zero naked predictions': { url: '/about/#standards', category: 'concept', defaultLabel: 'Zero Naked Predictions' },
  'chime': { url: '/papers/paper-e-frb-triage/', category: 'concept', defaultLabel: 'CHIME/FRB Catalog 1' },
  'quaia': { url: '/papers/paper-a-cosmic-dipole/', category: 'concept', defaultLabel: 'Quaia G20.5 Quasars' },
};

/**
 * Resolves a target string into a normalized URL, label, and category.
 */
export function resolveWikilink(target: string, customLabel?: string): { url: string; label: string; category: WikilinkMatch['category'] } {
  const cleanTarget = target.trim().toLowerCase();
  const match = WIKILINK_MAP[cleanTarget];

  if (match) {
    return {
      url: match.url,
      label: customLabel || match.defaultLabel || target,
      category: match.category,
    };
  }

  // Fallback: search for partial match
  for (const [key, val] of Object.entries(WIKILINK_MAP)) {
    if (cleanTarget.includes(key) || key.includes(cleanTarget)) {
      return {
        url: val.url,
        label: customLabel || val.defaultLabel || target,
        category: val.category,
      };
    }
  }

  // Default fallback link to search or matrix
  return {
    url: `/matrix/?q=${encodeURIComponent(target)}`,
    label: customLabel || target,
    category: 'concept',
  };
}

/**
 * Parses raw text or HTML, converting all `[[Target]]` and `[[Target|Label]]` instances
 * into structured semantic `<a class="wikilink">` anchors.
 */
export function parseWikilinks(content: string): string {
  if (!content || !content.includes('[[')) {
    return content;
  }

  const wikilinkRegex = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;

  return content.replace(wikilinkRegex, (_, target, customLabel) => {
    const resolved = resolveWikilink(target, customLabel);
    return `<a href="${resolved.url}" class="wikilink" data-wikilink="${encodeURIComponent(target)}" data-category="${resolved.category}" title="${resolved.label}">${resolved.label}</a>`;
  });
}
