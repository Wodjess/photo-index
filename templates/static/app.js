// Photo-index frontend: search + 5-card grid + cinema mode.
// No external deps. Vanilla JS. Module is loaded with `type="module"`.

const grid         = document.getElementById('grid');
const form         = document.getElementById('searchForm');
const input        = document.getElementById('q');
const cinema       = document.getElementById('cinema');
const cinemaImg    = document.getElementById('cinemaImg');
const cinemaCounter= document.getElementById('cinemaCounter');
const zoomLabel    = document.getElementById('zoomLabel');
const cinemaClose  = cinema.querySelector('.cinema-close');
const cinemaPrev   = cinema.querySelector('.cinema-prev');
const cinemaNext   = cinema.querySelector('.cinema-next');
const zoomInBtn    = cinema.querySelector('.zoom-in');
const zoomOutBtn   = cinema.querySelector('.zoom-out');
const zoomResetBtn = cinema.querySelector('.zoom-reset');
const fitToggleBtn = cinema.querySelector('.fit-toggle');
const cinemaStage  = cinema.querySelector('.cinema-stage');

const ICONS = {
  fit:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  actual: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>',
};

const FOCUSABLE_SELECTOR = 'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])';

// ── state ──────────────────────────────────────────────────────────────────
let results        = [];        // [{name, image_url, ...}, ...]
let ci             = -1;        // current cinema index (-1 = closed)
let isActual       = false;     // fit vs actual
let scale          = 1;
let tx             = 0;
let ty             = 0;
let lastTap        = 0;         // for double-tap
let pinchStart     = null;      // {d, scale, tx, ty, midX, midY, point:{x,y}}
let panStart       = null;      // {x, y, tx, ty, moved, time, isTap, tapTarget}
let pointers       = new Map();
let prevActiveEl   = null;      // saved focus before openCinema
let closeTapTimer  = null;      // for double-tap aware close-on-tap
let searchSeq      = 0;         // for race-free search (A9)
let navToken       = 0;         // for race-free cross-fade (A3)
let closeGen       = 0;         // 4-Round: invalidate in-flight close animations on re-open
let reduceMotion   = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── search ─────────────────────────────────────────────────────────────────
const K_DEFAULT = 5;          // matches SSR default_k in the template
const K_MIN = 1;
let K_MAX = 25;                 // re-assigned at settings() init from data-max-k (A40)

function clampK(n) {
  n = parseInt(n, 10);
  if (!Number.isFinite(n)) return K_DEFAULT;
  return Math.max(K_MIN, Math.min(K_MAX, n));
}

function getStoredK() {
  try {
    const v = parseInt(localStorage.getItem('photoindex.k') || '', 10);
    if (Number.isFinite(v)) return clampK(v);
  } catch (_) {}
  return K_DEFAULT;
}

function setStoredK(n) {
  try { localStorage.setItem('photoindex.k', String(clampK(n))); } catch (_) {}
}

async function doSearch(q) {
  const myToken = ++searchSeq;
  if (!q || !q.trim()) {
    renderEmpty();
    return;
  }
  fadeOutGrid(() => {
    if (myToken !== searchSeq) return;
    const k = getStoredK();
    fetch('/api/search?q=' + encodeURIComponent(q) + '&k=' + k, { headers: { 'Accept': 'application/json' } })
      .then((r) => {
        if (myToken !== searchSeq) return null;
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then((data) => {
        if (myToken !== searchSeq || !data) return;
        results = Array.isArray(data.results) ? data.results : [];
        if (results.length === 0) {
          renderEmpty();
        } else {
          renderGrid(results, true);
        }
      })
      .catch((err) => {
        if (myToken !== searchSeq) return;
        results = [];
        grid.innerHTML = '';
        grid.classList.add('is-ready');
        // A21: no console noise; UI feedback only.
      });
  });
}

function renderEmpty() {
  results = [];
  grid.innerHTML = '';
  grid.classList.add('is-ready');
}

function fadeOutGrid(done) {
  grid.classList.remove('is-ready');
  requestAnimationFrame(() => setTimeout(done, 220));
}

function renderGrid(items, fadeIn) {
  grid.innerHTML = '';
  for (let i = 0; i < items.length; i++) {
    const r = items[i];
    const card = document.createElement('article');
    card.className = 'card';
    card.setAttribute('data-idx', String(i));
    card.setAttribute('data-pos', String(i));
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', 'Изображение ' + (i + 1));
    const ph = document.createElement('div');
    ph.className = 'ph';
    const img = document.createElement('img');
    img.alt = '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.draggable = false; // 4-Round: suppress native image-drag ghost
    img.src = r.image_url; // server-trusted; do not escape (A21)
    card.appendChild(ph);
    card.appendChild(img);
    grid.appendChild(card);
  }
  wireBlurUp();
  if (fadeIn) {
    requestAnimationFrame(() => grid.classList.add('is-ready'));
  } else {
    grid.classList.add('is-ready');
  }
}

function wireBlurUp() {
  // Bug-2+4: when an image loads, also stamp the card's aspect-ratio to the
  // image's natural ratio, so all cards in the row share the same height
  // and width = height × naturalRatio. Square placeholder stays until then.
  grid.querySelectorAll('.card img').forEach((img) => {
    const card = img.closest('.card');
    const stamp = () => {
      img.classList.add('is-loaded');
      if (card && img.naturalWidth > 0 && img.naturalHeight > 0) {
        card.style.aspectRatio = img.naturalWidth + ' / ' + img.naturalHeight;
      }
    };
    if (img.complete && img.naturalWidth > 0) {
      stamp();
    } else {
      img.addEventListener('load',  stamp, { once: true });
      img.addEventListener('error', () => {
        img.removeAttribute('src');
        img.classList.add('is-loaded');
        // Keep the 1/1 fallback aspect on error so the layout doesn't break.
      }, { once: true });
    }
  });
}

// ── grid click → open cinema ──────────────────────────────────────────────
grid.addEventListener('click', (e) => {
  const card = e.target.closest('.card');
  if (!card) return;
  const idx = parseInt(card.getAttribute('data-idx') || '-1', 10);
  if (idx >= 0) openCinema(idx);
});
grid.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  const card = e.target.closest('.card');
  if (!card) return;
  e.preventDefault();
  const idx = parseInt(card.getAttribute('data-idx') || '-1', 10);
  if (idx >= 0) openCinema(idx);
});

// ── cinema mode ────────────────────────────────────────────────────────────
function getFocusableInCinema() {
  return Array.from(cinema.querySelectorAll(FOCUSABLE_SELECTOR));
}

function trapTab(e) {
  if (e.key !== 'Tab') return;
  const items = getFocusableInCinema();
  if (items.length === 0) return;
  const first = items[0];
  const last  = items[items.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

function openCinema(idx) {
  if (idx < 0 || idx >= results.length) return;
  // 4-Round Bug-1: invalidate any in-flight closeCinema from a previous
  // session so its 400ms setTimeout does not tear down the new open's DOM.
  ++closeGen;
  ci = idx;
  prevActiveEl = document.activeElement; // A11
  cinemaImg.draggable = false;            // 4-Round: suppress native drag ghost
  resetZoom();
  isActual = false;
  cinemaImg.classList.remove('is-actual');
  fitToggleBtn.innerHTML = ICONS.fit;
  cinemaCounter.textContent = (idx + 1) + ' / ' + results.length;
  cinemaImg.alt = 'Результат ' + (ci + 1); // A21

  // A2: choose open animation path based on reduce-motion.
  if (reduceMotion) {
    setCinemaImageSrc(results[idx].image_url);
    cinema.classList.add('is-open');
    cinema.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    cinemaClose.focus();
    return;
  }

  // FLIP open (A2): capture card rect, size the cinema img to it,
  // then transition to centered fullscreen.
  const card = grid.querySelector('.card[data-idx="' + idx + '"]');
  if (card) {
    const rect = card.getBoundingClientRect();
    cinemaImg.style.transition = 'none';
    cinemaImg.style.position = 'fixed';
    cinemaImg.style.left = rect.left + 'px';
    cinemaImg.style.top  = rect.top  + 'px';
    cinemaImg.style.width  = rect.width  + 'px';
    cinemaImg.style.height = rect.height + 'px';
    cinemaImg.style.maxWidth = '';
    cinemaImg.style.maxHeight = '';
    cinemaImg.style.objectFit = 'cover';
    cinemaImg.style.transform = 'translate(0,0) scale(1)';
    void cinemaImg.offsetWidth;
  }

  cinema.classList.add('is-open');
  cinema.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';

  // A10: wait for the new image to be decoded before starting the FLIP
  // (so we don't animate an empty box).
  setCinemaImageSrc(results[idx].image_url);
  startFlipOpen();
}

function startFlipOpen() {
  const afterDecode = () => {
    requestAnimationFrame(() => {
      cinemaImg.style.transition =
        'left '   + 350 + 'ms cubic-bezier(0.22, 1, 0.36, 1),' +
        'top '    + 350 + 'ms cubic-bezier(0.22, 1, 0.36, 1),' +
        'width '  + 350 + 'ms cubic-bezier(0.22, 1, 0.36, 1),' +
        'height ' + 350 + 'ms cubic-bezier(0.22, 1, 0.36, 1)';
      cinemaImg.style.left = '4vw';
      cinemaImg.style.top  = '7vh';
      cinemaImg.style.width  = '92vw';
      cinemaImg.style.height = '86vh';
      cinemaImg.style.objectFit = 'contain';
      setTimeout(() => {
        cinemaImg.style.transition = '';
        cinemaImg.style.position = '';
        cinemaImg.style.left = '';
        cinemaImg.style.top = '';
        cinemaImg.style.width = '';
        cinemaImg.style.height = '';
        cinemaImg.style.maxWidth = '92vw';
        cinemaImg.style.maxHeight = '86vh';
      }, 380);
    });
  };
  // A10: prefer decode() over load to avoid empty-box animation.
  if (cinemaImg.decode) {
    cinemaImg.decode().then(afterDecode).catch(afterDecode);
  } else {
    if (cinemaImg.complete) afterDecode();
    else cinemaImg.addEventListener('load', afterDecode, { once: true });
  }
}

function closeCinema() {
  if (ci < 0) return;
  const idx = ci;
  // 4-Round Bug-1: mark close in-flight so a re-open can invalidate the
  // pending 400ms cleanup if the user re-opens the cinema quickly.
  const myGen = ++closeGen;
  // 4-Round Bug-1: set ci = -1 immediately so any stray pointerup/click that
  // fires during the close animation cannot re-trigger closeCinema.
  ci = -1;
  // A7: clear pending close-tap timer (avoid race with double-tap).
  if (closeTapTimer) { clearTimeout(closeTapTimer); closeTapTimer = null; }

  // A2: reverse FLIP. If reduce-motion, just snap close.
  if (reduceMotion) {
    cinema.classList.remove('is-open');
    cinema.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    if (prevActiveEl && typeof prevActiveEl.focus === 'function') prevActiveEl.focus();
    return;
  }

  // Bug-5 fix: order is critical.
  //  1. Park cinemaImg at the current fullscreen rect with `position: fixed`
  //     and transition: none. This snapshot is our FLIP "from" rect.
  //  2. On the next frame, start the FLIP transition to the card's rect AND
  //     trigger the cinema's CSS transition to fade opacity/background out.
  //     Both run in parallel for the same duration.
  //  3. After 380ms, the cinema is fully transparent. NOW we can safely
  //     remove `is-open` and clear the inline styles — no flicker, because
  //     the cinema is invisible at that exact moment.
  const card = grid.querySelector('.card[data-idx="' + idx + '"]');
  const startRect = cinemaImg.getBoundingClientRect();
  cinemaImg.style.transition = 'none';
  cinemaImg.style.position = 'fixed';
  cinemaImg.style.left   = startRect.left + 'px';
  cinemaImg.style.top    = startRect.top  + 'px';
  cinemaImg.style.width  = startRect.width  + 'px';
  cinemaImg.style.height = startRect.height + 'px';
  cinemaImg.style.maxWidth = '';
  cinemaImg.style.maxHeight = '';
  cinemaImg.style.objectFit = 'cover';
  cinemaImg.style.transformOrigin = 'center center';
  void cinemaImg.offsetWidth;

  const cleanup = () => {
    if (myGen !== closeGen) return; // 4-Round: a re-open happened; let it own the DOM
    cinema.classList.remove('is-open');
    cinema.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    cinemaImg.style.transition = '';
    cinemaImg.style.position = '';
    cinemaImg.style.left = '';
    cinemaImg.style.top = '';
    cinemaImg.style.width = '';
    cinemaImg.style.height = '';
    cinemaImg.style.transformOrigin = '';
    cinemaImg.style.maxWidth = '92vw';
    cinemaImg.style.maxHeight = '86vh';
    cinemaImg.style.objectFit = '';
    resetZoom();
    if (prevActiveEl && typeof prevActiveEl.focus === 'function') prevActiveEl.focus();
  };

  if (!card) {
    // Fallback: simple fade close. Animate opacity/background of `.cinema`
    // to 0, then remove `is-open`.
    cinema.style.transition = 'background 280ms ease, opacity 280ms ease';
    cinema.style.opacity = '0';
    cinema.style.background = 'rgba(5, 7, 13, 0)';
    setTimeout(cleanup, 300);
    return;
  }
  const endRect = card.getBoundingClientRect();
  // Bring the source card above the dim overlay during the animation.
  card.style.visibility = 'visible';
  requestAnimationFrame(() => {
    if (myGen !== closeGen) return; // re-opened during open path
    cinemaImg.style.transition =
      'left '   + 380 + 'ms cubic-bezier(0.22, 1, 0.36, 1),' +
      'top '    + 380 + 'ms cubic-bezier(0.22, 1, 0.36, 1),' +
      'width '  + 380 + 'ms cubic-bezier(0.22, 1, 0.36, 1),' +
      'height ' + 380 + 'ms cubic-bezier(0.22, 1, 0.36, 1)';
    cinemaImg.style.left   = endRect.left + 'px';
    cinemaImg.style.top    = endRect.top  + 'px';
    cinemaImg.style.width  = endRect.width  + 'px';
    cinemaImg.style.height = endRect.height + 'px';
    // Fade the cinema overlay (background + opacity) in parallel so the
    // reverse-FLIP and the dim-out finish at the same moment. The flash
    // described in Bug-5 happened because we removed `is-open` BEFORE the
    // FLIP transition finished, exposing the un-styled cinemaImg for one
    // frame. Now `is-open` stays on until 380ms, opacity goes to 0 first.
    cinema.style.transition = 'background 380ms ease, opacity 380ms ease';
    cinema.style.opacity = '0';
    cinema.style.background = 'rgba(5, 7, 13, 0)';
    setTimeout(() => {
      if (myGen !== closeGen) return; // re-opened during close
      // Reset the inline cinema transition so the next open is clean.
      cinema.style.transition = '';
      cinema.style.opacity = '';
      cinema.style.background = '';
      cleanup();
    }, 400);
  });
}

function setCinemaImageSrc(url) {
  // A3 + A6: token-guarded, error-safe.
  const myToken = ++navToken;
  cinemaImg.classList.add('is-fading');
  // Defer src swap by 1 frame so the fade-out is visible.
  setTimeout(() => {
    if (myToken !== navToken) return;
    const onSettled = () => {
      if (myToken !== navToken) return;
      cinemaImg.classList.remove('is-fading');
    };
    cinemaImg.addEventListener('load',  onSettled, { once: true });
    cinemaImg.addEventListener('error', onSettled, { once: true }); // A6
    cinemaImg.src = url;
    if (cinemaImg.complete && cinemaImg.naturalWidth > 0) onSettled();
  }, 140);
}

function navCinema(delta) {
  if (ci < 0 || results.length === 0) return;
  // 4-Round Bug-1: cancel any pending close-tap from a stray single tap
  // so it cannot fire and close the modal mid-nav.
  if (closeTapTimer) { clearTimeout(closeTapTimer); closeTapTimer = null; }
  const next = (ci + delta + results.length) % results.length;
  if (next === ci) return;
  ci = next;
  resetZoom();
  isActual = false;
  cinemaImg.classList.remove('is-actual');
  fitToggleBtn.innerHTML = ICONS.fit;
  cinemaCounter.textContent = (ci + 1) + ' / ' + results.length;
  cinemaImg.alt = 'Результат ' + (ci + 1);
  setCinemaImageSrc(results[ci].image_url);
}

function resetZoom() {
  scale = 1; tx = 0; ty = 0;
  // 4-Round Bug-3: in is-actual mode the image is at natural pixel size, so
  // resetting transform alone leaves a huge overflowing image. Exit is-actual
  // here so "reset to 100%" always re-centers inside the stage.
  if (isActual) {
    isActual = false;
    cinemaImg.classList.remove('is-actual');
    cinemaImg.style.maxWidth = '92vw';
    cinemaImg.style.maxHeight = '86vh';
    fitToggleBtn.innerHTML = ICONS.fit;
  }
  cinemaImg.style.transform = '';
  applyTransform();
  zoomLabel.textContent = '100%';
}

function applyTransform() {
  cinemaImg.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
  zoomLabel.textContent = Math.round(scale * 100) + '%';
}

function zoomBy(factor, cx, cy) {
  const newScale = Math.max(1, Math.min(4, scale * factor));
  if (newScale === scale) return; // no-op
  // 4-Round+ Bug-3b: when the user zooms out all the way to 100%, snap the
  // image back to the center instantly. Any pan offset from the zoomed-in
  // state is meaningless at scale=1 (the image is fully contained inside
  // the stage via flex-center, so an offset would leave it mis-centered).
  if (newScale === 1) {
    scale = 1;
    tx = 0;
    ty = 0;
    applyTransform();
    return;
  }
  const rect = cinemaImg.getBoundingClientRect();
  // rect is post-transform: it already includes translate(tx, ty).
  // Image-space point under (cx, cy) at the OLD scale:
  const ox = (cx - (rect.left + rect.width / 2)) / scale;
  const oy = (cy - (rect.top  + rect.height / 2)) / scale;
  scale = newScale;
  // We want naturalCenterX + tx_new + ox * newScale === cx  (and same for y).
  // So tx_new = cx - ox * newScale - naturalCenterX.
  // But naturalCenterX = rect.left + rect.width/2 - tx  (rect = nat + tx).
  // Hence tx_new = cx - ox * newScale - (rect.left + rect.width/2 - tx).
  // And tx += (tx_new - tx) = (cx - ox * newScale) - (rect.left + rect.width/2).
  const newCenterX = cx - ox * scale;
  const newCenterY = cy - oy * scale;
  const curCenterX = rect.left + rect.width  / 2;
  const curCenterY = rect.top  + rect.height / 2;
  tx += (newCenterX - curCenterX);
  ty += (newCenterY - curCenterY);
  applyTransform();
}

// ── cinema event wiring ────────────────────────────────────────────────────
// Bug-1 defensive: stop pointer events on every control so they can never
// reach cinemaStage's pan/close-tap logic. 4-Round: also stopImmediate
// + preventDefault on pointerdown so the control fully owns the gesture.
function stopAll(e) {
  e.stopPropagation();
  e.stopImmediatePropagation();
}
[cinemaClose, cinemaPrev, cinemaNext, zoomInBtn, zoomOutBtn, zoomResetBtn, fitToggleBtn].forEach((btn) => {
  btn.addEventListener('pointerdown', stopAll);
  btn.addEventListener('pointerup',   stopAll);
});
// 4-Round Bug-1: any click on a control fully owns the gesture and must not
// leak to the stage. We also clear pending close-timers defensively.
function ctrlClick(action) {
  return (e) => {
    e.stopPropagation();
    e.stopImmediatePropagation();
    if (closeTapTimer) { clearTimeout(closeTapTimer); closeTapTimer = null; }
    action(e);
  };
}
cinemaClose.addEventListener('click',  ctrlClick(() => closeCinema()));
cinemaPrev.addEventListener('click',   ctrlClick(() => navCinema(-1)));
cinemaNext.addEventListener('click',   ctrlClick(() => navCinema(+1)));
zoomInBtn.addEventListener('click',    ctrlClick(() => zoomBy(1.25, window.innerWidth/2, window.innerHeight/2)));
zoomOutBtn.addEventListener('click',   ctrlClick(() => zoomBy(1 / 1.25, window.innerWidth/2, window.innerHeight/2)));
zoomResetBtn.addEventListener('click', ctrlClick(() => resetZoom()));
fitToggleBtn.addEventListener('click', ctrlClick(() => {
  isActual = !isActual;
  cinemaImg.classList.toggle('is-actual', isActual);
  // A8: clear or restore inline maxes so .is-actual actually wins.
  if (isActual) {
    cinemaImg.style.maxWidth = '';
    cinemaImg.style.maxHeight = '';
  } else {
    cinemaImg.style.maxWidth = '92vw';
    cinemaImg.style.maxHeight = '86vh';
    resetZoom();
  }
  fitToggleBtn.innerHTML = isActual ? ICONS.actual : ICONS.fit;
}));

// A22: drop the redundant cinema click handler. Stage pointerup is the
// real background-close path (with double-tap awareness — A7).

// Keyboard
document.addEventListener('keydown', (e) => {
  if (ci < 0) return;
  // A5: ignore when focus is in a text-editing element.
  const t = e.target;
  const tag = t && t.tagName;
  const inText = tag === 'INPUT' || tag === 'TEXTAREA' || (t && t.isContentEditable);
  trapTab(e);
  if (e.key === 'Escape') { e.preventDefault(); closeCinema(); return; }
  if (inText) return;
  if (e.key === 'ArrowLeft')  { e.preventDefault(); navCinema(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); navCinema(+1); }
  else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomBy(1.25, window.innerWidth/2, window.innerHeight/2); }
  else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomBy(1/1.25, window.innerWidth/2, window.innerHeight/2); }
  else if (e.key === '0') { e.preventDefault(); resetZoom(); }
});

// A16: plain wheel zoom (no modifier required).
cinema.addEventListener('wheel', (e) => {
  if (ci < 0) return;
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
  zoomBy(factor, e.clientX, e.clientY);
}, { passive: false });

// Pointer events: pan, pinch, swipe (A4), double-tap (A7), single-tap close.
let swipeArmed = false;
let swipeStartX = 0;
let swipeStartY = 0;

cinemaStage.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.cinema-btn') || e.target.closest('.cinema-toolbar') || e.target.closest('.cinema-counter') || e.target.closest('.cinema-close')) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) {
    panStart = {
      x: e.clientX, y: e.clientY, tx, ty,
      moved: false,
      time: Date.now(),
      isTap: true,
    };
    swipeArmed = true;
    swipeStartX = e.clientX;
    swipeStartY = e.clientY;
  } else if (pointers.size === 2) {
    // A19: record image point under the pinch midpoint.
    const pts = Array.from(pointers.values());
    const dx = pts[0].x - pts[1].x;
    const dy = pts[0].y - pts[1].y;
    const d  = Math.hypot(dx, dy);
    const midX = (pts[0].x + pts[1].x) / 2;
    const midY = (pts[0].y + pts[1].y) / 2;
    const rect = cinemaImg.getBoundingClientRect();
    const ox = (midX - (rect.left + rect.width / 2)) / scale;
    const oy = (midY - (rect.top  + rect.height / 2)) / scale;
    pinchStart = { d, scale, tx, ty, ox, oy, midX, midY };
    panStart = null;
    swipeArmed = false;
  }
  try { cinemaStage.setPointerCapture(e.pointerId); } catch (_) {}
});

cinemaStage.addEventListener('pointermove', (e) => {
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pinchStart && pointers.size === 2) {
    const pts = Array.from(pointers.values());
    const dx = pts[0].x - pts[1].x;
    const dy = pts[0].y - pts[1].y;
    const d  = Math.hypot(dx, dy);
    const factor = d / pinchStart.d;
    scale = Math.max(1, Math.min(4, pinchStart.scale * factor));
    // A19: keep image point under the new midpoint.
    const midX = (pts[0].x + pts[1].x) / 2;
    const midY = (pts[0].y + pts[1].y) / 2;
    const rect = cinemaImg.getBoundingClientRect();
    const newCenterX = midX - pinchStart.ox * scale;
    const newCenterY = midY - pinchStart.oy * scale;
    tx = newCenterX - (rect.left + rect.width / 2);
    ty = newCenterY - (rect.top  + rect.height / 2);
    applyTransform();
    swipeArmed = false;
  } else if (panStart && pointers.size === 1) {
    const dx = e.clientX - panStart.x;
    const dy = e.clientY - panStart.y;
    if (Math.hypot(dx, dy) > 4) panStart.moved = true;
    if (panStart.moved) {
      panStart.isTap = false;
      // A4: horizontal swipe detection (only at default zoom).
      if (scale <= 1.001 && swipeArmed) {
        if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) {
          navCinema(dx > 0 ? -1 : +1);
          swipeArmed = false;
          // reset pan state
          panStart.x = e.clientX;
          panStart.y = e.clientY;
          panStart.tx = tx;
          panStart.ty = ty;
          return;
        }
      }
      // Pan only when zoomed in.
      if (scale > 1) {
        tx = panStart.tx + dx;
        ty = panStart.ty + dy;
        applyTransform();
      }
    }
  }
});

cinemaStage.addEventListener('pointerup', (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinchStart = null;
  if (pointers.size === 0) {
    if (panStart && !panStart.moved) {
      const now = Date.now();
      const tapX = e.clientX;
      const tapY = e.clientY;
      const onControl = e.target.closest(
        '.cinema-btn, .cinema-toolbar, .cinema-counter, .cinema-close, .cinema-prev, .cinema-next'
      );
      // 4-Round Bug-1: defensively guard against ci having been set to -1 by
      // a parallel closeCinema (e.g. from a stray keyboard Escape). If
      // cinema is already closing, do not schedule another close-tap.
      if (ci < 0) {
        // already closed; ignore
      } else if (onControl) {
        // Tap landed on a control; let the control's own click handler deal
        // with it. No close-tap scheduled.
      } else if (now - lastTap < 300) {
        // A18: pass tap coords to zoomBy for double-tap zoom-in pivot.
        // Double-tap: toggle zoom. Only if the user double-tapped the image,
        // not the stage background.
        if (scale > 1.05) {
          resetZoom();
        } else {
          zoomBy(2.2, tapX, tapY);
        }
        lastTap = 0;
        if (closeTapTimer) { clearTimeout(closeTapTimer); closeTapTimer = null; }
      } else {
        // Single tap on empty stage: schedule close, but give 300ms for a
        // second tap to cancel (A7). A click on the image itself is treated
        // as a candidate close too — same UX, just zoom-in on second tap.
        lastTap = now;
        if (closeTapTimer) clearTimeout(closeTapTimer);
        closeTapTimer = setTimeout(() => {
          closeTapTimer = null;
          if (ci < 0) return; // 4-Round: race guard
          closeCinema();
        }, 300);
      }
    }
    panStart = null;
    swipeArmed = false;
  }
});

cinemaStage.addEventListener('pointercancel', () => {
  pointers.clear();
  pinchStart = null;
  panStart   = null;
  swipeArmed = false;
});

// A20: cleanup on pagehide.
window.addEventListener('pagehide', () => { document.body.style.overflow = ''; });

// ── form submit ────────────────────────────────────────────────────────────
form.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = input.value;
  history.replaceState(null, '', q ? '?q=' + encodeURIComponent(q) : location.pathname);
  if (ci >= 0) closeCinema();
  doSearch(q);
});

// ── boot ───────────────────────────────────────────────────────────────────
(function boot() {
  // A12: if SSR produced cards, wire blur-up on their existing <img> and
  // do NOT rebuild. Otherwise render from JSON.
  const initial = Array.isArray(window.__INITIAL_RESULTS__) ? window.__INITIAL_RESULTS__ : [];
  if (initial.length > 0) {
    results = initial;
    const ssrCards = grid.querySelectorAll('.card');
    if (ssrCards.length === initial.length) {
      // Trust SSR DOM; just attach blur-up.
      wireBlurUp();
      grid.classList.add('is-ready');
    } else {
      renderGrid(initial, false);
    }
  } else {
    grid.classList.add('is-ready');
  }
  setTimeout(() => { try { input.focus({ preventScroll: true }); } catch (_) { input.focus(); } }, 50);
})();

// ── Settings (Round 0) ─────────────────────────────────────────────────────
// Minimal, self-contained. Owns its own state. Plays nice with the
// existing cinema / search modules (no global state shared besides
// localStorage 'photoindex.k' which doSearch also reads).
(function settings() {
  const btn        = document.getElementById('settingsBtn');
  const backdrop   = document.getElementById('settingsBackdrop');
  const modal      = document.getElementById('settingsModal');
  const closeBtn   = document.getElementById('closeSettingsBtn');
  const dropZone   = document.getElementById('dropZone');
  const fileInput  = document.getElementById('fileInput');
  const fileList   = document.getElementById('fileList');
  const dropLimits = document.getElementById('dropLimits');
  const clearBtn   = document.getElementById('clearFilesBtn');
  const uploadBtn  = document.getElementById('uploadBtn');
  const statusBox  = document.getElementById('uploadStatus');
  const kRange     = document.getElementById('kRange');
  const kInput     = document.getElementById('kInput');

  if (!btn || !modal) return;

  // ── A25/A52 fix: read server-supplied limits from data-* attributes
  // on the modal (rendered by the template). Fall back to safe defaults.
  const _n = (v, d) => { const x = parseInt(v, 10); return Number.isFinite(x) ? x : d; };
  const MAX_FILES = _n(modal.dataset.maxFiles, 300);
  const MAX_BYTES = _n(modal.dataset.maxBytes, 200 * 1024 * 1024);
  // A40 fix: K_MAX is now server-driven.
  K_MAX = _n(modal.dataset.maxK, 25);

  // ── State ──
  let pendingFiles = [];   // File[]
  let lastFocus    = null;
  let pollTimer    = null;

  // ── A17 fix: while the modal is open, swallow stray drag/drop on the
  // page background. Without this, dropping a file outside the dropZone
  // navigates the browser to file://. We bind on document so the entire
  // viewport is covered; we do NOT stop the dropZone's own handlers.
  function _onDocDragOver(e) { if (!e.target.closest('.drop-zone')) e.preventDefault(); }
  function _onDocDrop(e)     { if (!e.target.closest('.drop-zone')) e.preventDefault(); }
  function _bindDocGuard(on) {
    if (on) {
      document.addEventListener('dragover', _onDocDragOver);
      document.addEventListener('drop',     _onDocDrop);
    } else {
      document.removeEventListener('dragover', _onDocDragOver);
      document.removeEventListener('drop',     _onDocDrop);
    }
  }

  // ── Open / close ──
  function open() {
    lastFocus = document.activeElement;
    btn.setAttribute('aria-expanded', 'true');
    backdrop.classList.add('is-open');
    backdrop.setAttribute('aria-hidden', 'false');
    modal.classList.add('is-open');
    modal.setAttribute('aria-hidden', 'false');
    _bindDocGuard(true);
    // Focus the drop-zone on next frame so the transition can run first.
    requestAnimationFrame(() => { try { dropZone.focus({ preventScroll: true }); } catch (_) { dropZone.focus(); } });
  }
  function close() {
    btn.setAttribute('aria-expanded', 'false');
    backdrop.classList.remove('is-open');
    backdrop.setAttribute('aria-hidden', 'true');
    modal.classList.remove('is-open');
    modal.setAttribute('aria-hidden', 'true');
    _bindDocGuard(false);
    // A12: stop the background poll loop when the user closes the modal.
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    if (lastFocus && typeof lastFocus.focus === 'function') {
      try { lastFocus.focus({ preventScroll: true }); } catch (_) { lastFocus.focus(); }
    }
  }

  btn.addEventListener('click', open);
  closeBtn.addEventListener('click', close);
  backdrop.addEventListener('click', close);

  // Keyboard: Escape closes, Tab is trapped.
  document.addEventListener('keydown', (e) => {
    if (!modal.classList.contains('is-open')) return;
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab') return;
    const focusables = modal.querySelectorAll('button, [tabindex="0"], input, select, textarea');
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last  = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  });

  // ── k slider + input ──
  // A23 fix: only clamp on 'change' (blur/Enter) for the number input.
  // Mid-typing on 'input' would snap "30" to "25" while the user is
  // still typing. The range is bound to 'input' because the range's
  // 'change' fires only on release; we still update the number input
  // on 'input' to keep both controls in sync visually.
  function syncK(src) {
    const n = clampK(src.value);
    kRange.value = n;
    kInput.value = n;
    setStoredK(n);
  }
  kRange.addEventListener('input',  () => syncK(kRange));
  kRange.addEventListener('change', () => syncK(kRange));
  kInput.addEventListener('change', () => syncK(kInput));
  kInput.addEventListener('blur',    () => syncK(kInput));
  // Initial value from localStorage.
  { const k = getStoredK(); kRange.value = k; kInput.value = k; }

  // ── File picker ──
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  ;['dragenter', 'dragover'].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('is-drag'); })
  );
  ;['dragleave', 'drop'].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('is-drag'); })
  );
  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (!dt) return;
    addFiles(dt.files);
  });
  fileInput.addEventListener('change', () => {
    addFiles(fileInput.files);
    // Reset so the same file can be picked again.
    fileInput.value = '';
  });

  function addFiles(list) {
    const incoming = Array.from(list || []);
    for (const f of incoming) {
      if (pendingFiles.length >= MAX_FILES) break;
      // Skip duplicates by name+size+lastModified.
      const sig = f.name + '|' + f.size + '|' + f.lastModified;
      const dupe = pendingFiles.some((g) => (g.name + '|' + g.size + '|' + g.lastModified) === sig);
      if (dupe) continue;
      pendingFiles.push(f);
    }
    renderFileList();
  }
  function clearFiles() {
    pendingFiles = [];
    renderFileList();
    setStatus('', null);
  }
  clearBtn.addEventListener('click', clearFiles);

  function totalBytes() {
    return pendingFiles.reduce((s, f) => s + (f.size || 0), 0);
  }

  function fmtBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
    return (b / (1024 * 1024)).toFixed(2) + ' MB';
  }

  function renderFileList() {
    if (pendingFiles.length === 0) {
      fileList.innerHTML = '';
      clearBtn.style.display = 'none';
      uploadBtn.disabled = true;
      return;
    }
    clearBtn.style.display = '';
    uploadBtn.disabled = (totalBytes() > MAX_BYTES);
    const rows = pendingFiles.slice(0, 50).map((f) => {
      return '<div class="file-row"><span class="name"></span><span class="bytes"></span></div>';
    }).join('');
    fileList.innerHTML = rows +
      (pendingFiles.length > 50
        ? '<div class="file-row"><span class="name">… ещё ' + (pendingFiles.length - 50) + '</span><span></span></div>'
        : '');
    // Fill names+bytes safely (textContent, not innerHTML).
    const items = fileList.querySelectorAll('.file-row');
    pendingFiles.slice(0, 50).forEach((f, i) => {
      const r = items[i];
      if (!r) return;
      r.querySelector('.name').textContent = f.name;
      r.querySelector('.bytes').textContent = fmtBytes(f.size || 0);
    });
    // Edge case: if size cap is exceeded, show a hint.
    if (totalBytes() > MAX_BYTES) {
      setStatus('Слишком большой размер: ' + fmtBytes(totalBytes()) + ' (лимит ' + fmtBytes(MAX_BYTES) + '). Удалите часть файлов.', 'err');
      uploadBtn.disabled = true;
    } else if (pendingFiles.length >= MAX_FILES) {
      setStatus('Достигнут лимит в ' + MAX_FILES + ' файлов.', 'err');
      uploadBtn.disabled = true;
    } else {
      setStatus('Готово к загрузке: ' + pendingFiles.length + ' файлов, ' + fmtBytes(totalBytes()) + '.', null);
    }
  }

  function setStatus(text, kind) {
    if (!text) {
      statusBox.hidden = true;
      statusBox.className = 'upload-status';
      statusBox.innerHTML = '';
      return;
    }
    statusBox.hidden = false;
    statusBox.className = 'upload-status' + (kind ? ' is-' + kind : '');
    statusBox.textContent = text;
  }

  // ── A1/A2-1 fix: setStatusHtml accepts an array of nodes + strings.
  // Strings are inserted as textContent (XSS-safe); DOM nodes appended directly.
  function setStatusHtml(parts, kind) {
    statusBox.hidden = false;
    statusBox.className = 'upload-status' + (kind ? ' is-' + kind : '');
    statusBox.innerHTML = '';
    for (const p of parts) {
      if (p == null) continue;
      if (typeof p === 'string' || typeof p === 'number') {
        statusBox.appendChild(document.createTextNode(String(p)));
      } else if (p instanceof Node) {
        statusBox.appendChild(p);
      }
    }
  }
  function statusWithJob(text, jobId, kind) {
    setStatusHtml([text, ' ', (() => { const c = document.createElement('code'); c.textContent = jobId; return c; })()], kind);
  }

  // ── Upload ──
  uploadBtn.addEventListener('click', uploadNow);

  async function uploadNow() {
    if (pendingFiles.length === 0 || totalBytes() > MAX_BYTES) return;
    setStatus('Загружаем ' + pendingFiles.length + ' файлов…', null);
    uploadBtn.disabled = true;
    clearBtn.disabled  = true;

    const fd = new FormData();
    pendingFiles.forEach((f) => fd.append('files', f, f.name));

    let resp;
    try {
      resp = await fetch('/api/upload', { method: 'POST', body: fd });
    } catch (e) {
      setStatus('Ошибка сети: ' + (e && e.message || e), 'err');
      uploadBtn.disabled = false;
      clearBtn.disabled  = false;
      return;
    }
    if (!resp.ok) {
      let msg = 'HTTP ' + resp.status;
      try { const j = await resp.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
      setStatus('Ошибка: ' + msg, 'err');
      uploadBtn.disabled = false;
      clearBtn.disabled  = false;
      return;
    }
    let data;
    try { data = await resp.json(); } catch (_) { data = null; }
    const jobId = data && data.job_id;
    const count = data && data.count;
    if (!jobId) {
      setStatus('Сервер не вернул job_id.', 'err');
      uploadBtn.disabled = false;
      clearBtn.disabled  = false;
      return;
    }
    setStatus('Принято: ' + count + ' файлов. Очередь: ' + jobId, 'ok');
    pendingFiles = [];
    renderFileList();
    clearBtn.disabled = false;
    pollJob(jobId);
  }

  async function pollJob(jobId) {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    const tick = async () => {
      let resp;
      try { resp = await fetch('/api/upload/' + encodeURIComponent(jobId)); }
      catch (e) { pollTimer = setTimeout(tick, 3000); return; }
      if (!resp.ok) {
        // A22: retry instead of giving up. 404 (job expired) is terminal.
        if (resp.status === 404) {
          setStatus('Статус: задача не найдена (истекла?).', 'err');
          return;
        }
        setStatus('Статус: ' + resp.status + ', повтор через 3с…', 'err');
        pollTimer = setTimeout(tick, 3000);
        return;
      }
      let j;
      try { j = await resp.json(); } catch (_) { j = null; }
      if (!j) { pollTimer = setTimeout(tick, 3000); return; }
      const total = j.total || 0;
      const done  = j.done  || 0;
      const failed = j.failed || 0;
      const status = j.status || 'unknown';
      // A1: use setStatusHtml so the jobId renders as a real <code> element
      // (the old setStatus(... <code></code> ...) wrote a literal "<code>"
      // string via textContent, which then broke querySelector('code')).
      const kind = (status === 'done') ? 'ok' : (status === 'failed' ? 'err' : null);
      setStatusHtml([
        'Статус: ', status, ' · готово ', done, '/', total,
        failed ? [' · ошибок ', failed] : null,
        ' · job ',
        (() => { const c = document.createElement('code'); c.textContent = jobId; return c; })(),
      ], kind);
      if (status === 'done' || status === 'failed') return;
      pollTimer = setTimeout(tick, 2000);
    };
    tick();
  }
})();

/* ── Delete modal ─────────────────────────────────────────────────── */
(function deleteModal() {
  const openBtn     = document.getElementById('deleteModeBtn');
  const backdrop    = document.getElementById('deleteBackdrop');
  const modal       = document.getElementById('deleteModal');
  const closeBtn    = document.getElementById('closeDeleteBtn');
  const searchInput = document.getElementById('deleteSearch');
  const searchBtn   = document.getElementById('deleteSearchBtn');
  const resultsDiv  = document.getElementById('deleteResults');
  const statusDiv   = document.getElementById('deleteStatus');

  if (!openBtn || !modal || !backdrop) return;

  function show() {
    modal.setAttribute('aria-hidden', 'false');
    backdrop.setAttribute('aria-hidden', 'false');
    backdrop.style.zIndex = '100';
    modal.style.zIndex = '110';
    backdrop.style.opacity = '1';
    backdrop.style.pointerEvents = 'auto';
    modal.style.opacity = '1';
    modal.style.pointerEvents = 'auto';
    modal.style.transform = 'translate(-50%, -50%) scale(1)';
    searchInput.value = '';
    resultsDiv.innerHTML = '';
    statusDiv.hidden = true;
    searchInput.focus();
  }
  function hide() {
    modal.setAttribute('aria-hidden', 'true');
    backdrop.setAttribute('aria-hidden', 'true');
    backdrop.style.opacity = '0';
    backdrop.style.pointerEvents = 'none';
    modal.style.opacity = '0';
    modal.style.pointerEvents = 'none';
    modal.style.transform = 'translate(-50%, -48%) scale(0.97)';
  }

  openBtn.addEventListener('click', show);
  closeBtn.addEventListener('click', hide);
  backdrop.addEventListener('click', hide);

  function setStatus(text, kind) {
    statusDiv.hidden = false;
    statusDiv.textContent = text;
    statusDiv.className = 'upload-status' + (kind ? ' ' + kind : '');
  }

  async function doSearch() {
    const q = searchInput.value.trim();
    if (!q) return;
    resultsDiv.innerHTML = '<div style="grid-column:1/-1;text-align:center;color:var(--text-muted)">Поиск...</div>';
    try {
      const k = parseInt(document.getElementById('kRange')?.value || '5', 10);
      const resp = await fetch('/api/search?q=' + encodeURIComponent(q) + '&k=' + k);
      const data = await resp.json();
      renderResults(data.results || []);
    } catch (e) {
      resultsDiv.innerHTML = '<div style="grid-column:1/-1;color:#ff6b6b">Ошибка поиска</div>';
    }
  }

  function renderResults(results) {
    resultsDiv.innerHTML = '';
    if (!results.length) {
      resultsDiv.innerHTML = '<div style="grid-column:1/-1;text-align:center;color:var(--text-muted)">Ничего не найдено</div>';
      return;
    }
    for (const r of results) {
      const card = document.createElement('div');
      card.className = 'delete-card';

      const img = document.createElement('img');
      img.src = r.image_url || '/image/' + encodeURIComponent(r.name);
      img.alt = r.name;
      img.loading = 'lazy';

      const info = document.createElement('div');
      info.className = 'delete-card-info';
      info.textContent = r.name;
      info.title = r.name;

      const actions = document.createElement('div');
      actions.className = 'delete-card-actions';

      const delBtn = document.createElement('button');
      delBtn.className = 'btn-danger-sm';
      delBtn.textContent = 'Удалить';
      delBtn.addEventListener('click', () => doDelete(r.name, card));

      actions.appendChild(delBtn);
      card.appendChild(img);
      card.appendChild(info);
      card.appendChild(actions);
      resultsDiv.appendChild(card);
    }
  }

  async function doDelete(name, card) {
    if (!confirm('Удалить «' + name + '»? Это действие необратимо.')) return;
    try {
      const resp = await fetch('/api/image/' + encodeURIComponent(name), { method: 'DELETE' });
      const data = await resp.json();
      if (resp.ok) {
        card.remove();
        setStatus('Удалено: ' + name + ' (осталось ' + data.remaining + ')', 'ok');
      } else {
        setStatus('Ошибка: ' + (data.detail || 'unknown'), 'err');
      }
    } catch (e) {
      setStatus('Ошибка сети', 'err');
    }
  }

  searchBtn.addEventListener('click', doSearch);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doSearch();
  });
})();
