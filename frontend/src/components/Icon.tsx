// Line icons on a 24 px grid, drawn in the current text colour.
const PATHS = {
  logo: "M4 17l5-6 4 3 7-8",
  scan: "M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M7.5 12h9",
  overview: "M4 4h6v7H4zM14 4h6v4h-6zM14 12h6v8h-6zM4 15h6v5H4z",
  quality: "M12 3l8 3v5c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6zM8.5 12l2.5 2.5 4.5-5",
  list: "M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01",
  pattern: "M5 4h5v5H5zM14 15h5v5h-5zM10 6.5h6.5V15M14 17.5H7.5V9",
  model: "M9 3h6M10 3v6l-5 10a1.5 1.5 0 001.4 2h11.2a1.5 1.5 0 001.4-2L14 9V3M7.5 15h9",
  cluster: "M9.5 7a3 3 0 11-6 0 3 3 0 016 0zM20.5 8a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0zM16 17a3.5 3.5 0 11-7 0 3.5 3.5 0 017 0z",
  network: "M8 5a3 3 0 11-6 0 3 3 0 016 0zM22 7a3 3 0 11-6 0 3 3 0 016 0zM14 19a3 3 0 11-6 0 3 3 0 016 0zM8 5.5l8 1M6.5 7.5l3.5 9M17.5 9.5L13 16.5",
  gauge: "M4 18a9 9 0 1116 0M12 14l4-5M12 14h.01",
  cloud: "M7 19h11a4 4 0 00.5-8 6.5 6.5 0 00-12.5-1A4.6 4.6 0 007 19z",
  menu: "M4 7h16M4 12h16M4 17h16",
  file: "M14 3H6v18h12V7zM14 3v4h4M9 12h6M9 16h6",
  data: "M20 6c0 1.7-3.6 3-8 3S4 7.7 4 6s3.6-3 8-3 8 1.3 8 3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  attack: "M12 4l9 16H3zM12 10v4M12 17h.01",
  download: "M12 4v11M7.5 10.5L12 15l4.5-4.5M4 20h16",
};
export type IconName = keyof typeof PATHS;

export function Icon({ name }: { name: IconName }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d={PATHS[name]} />
    </svg>
  );
}
