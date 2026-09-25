import { defineConfig } from 'astro/config';
import rehypeSidenotes from './src/plugins/rehype-sidenotes.ts';

export default defineConfig({
  site: 'https://astro.subsurfaces.net',
  markdown: {
    rehypePlugins: [rehypeSidenotes],
  },
});
