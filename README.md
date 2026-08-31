# Michael's Portfolio

A fast, editorial-style personal portfolio and blog built with Astro for static-first delivery, strong page performance, and a small runtime surface.

## Build

```bash
pnpm install
pnpm run build
```

The build outputs the deployable site to `dist/`. Visitor geography is generated server-side from Nginx logs as privacy-preserving daily aggregates; `/visitor-insights/` is private and must be protected by Nginx authentication.

## License

- Source code and scripts: MIT License.
- Blog posts and documentation: CC BY 4.0.

See [LICENSE](LICENSE) for the full terms.

## Getting Started

1. Install dependencies: `pnpm install`
2. Run development server: `pnpm run dev`
3. Build for production: `pnpm run build`
