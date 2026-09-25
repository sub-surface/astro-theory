import { defineConfig } from 'astro/config';
import rehypeSidenotes from './src/plugins/rehype-sidenotes.ts';

export default defineConfig({
  markdown: {
    rehypePlugins: [rehypeSidenotes],
  },
});
