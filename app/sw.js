// Offline support: the app shell is cached at install, the pose model and MediaPipe runtime the
// first time they are used. Bump VERSION when the app changes so phones pick up the new files.
const VERSION = 'v3';
const SHELL = [
  './', 'index.html', 'styles.css', 'manifest.webmanifest',
  'js/app.js', 'js/classic.js', 'js/estimator.js', 'js/geometry.js', 'js/holds.js',
  'js/pictograms.js', 'js/pipeline.js', 'js/references.js', 'js/runner.js', 'js/select.js', 'js/video.js',
  'icons/icon.svg', 'icons/icon-180.png', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(`shell-${VERSION}`).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('shell-') && k !== `shell-${VERSION}`).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== location.origin) return;
  const vendor = url.pathname.includes('/vendor/');
  const cacheName = vendor ? 'vendor' : `shell-${VERSION}`;
  event.respondWith(
    caches.open(cacheName).then(async (cache) => {
      const hit = await cache.match(event.request, { ignoreSearch: true });
      if (hit) return hit;
      const response = await fetch(event.request);
      if (response.ok && (vendor || SHELL.some((p) => url.pathname.endsWith(p.replace('./', '/'))))) {
        cache.put(event.request, response.clone());
      }
      return response;
    }),
  );
});
