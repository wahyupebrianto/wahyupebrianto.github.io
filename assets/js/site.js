/* Site behaviour: theme toggle, mobile menu, sub-menu.  No dependencies. */
(function () {
  'use strict';

  var root = document.documentElement;

  /* --- Theme toggle ------------------------------------------------------ */
  var themeBtn = document.querySelector('.theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) { /* private mode */ }
    });
  }

  /* --- Mobile menu ------------------------------------------------------- */
  var navBtn = document.querySelector('.nav-toggle');
  var nav = document.getElementById('site-nav');
  function setNav(open) {
    if (!navBtn || !nav) return;
    navBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    nav.classList.toggle('open', open);
  }
  if (navBtn && nav) {
    navBtn.addEventListener('click', function () {
      setNav(navBtn.getAttribute('aria-expanded') !== 'true');
    });
  }

  /* --- Sub-menus (touch + keyboard; hover is handled in CSS) ------------- */
  var toggles = document.querySelectorAll('.sub-toggle');
  function closeSubs(except) {
    document.querySelectorAll('.has-sub.open').forEach(function (li) {
      if (li !== except) {
        li.classList.remove('open');
        var b = li.querySelector('.sub-toggle');
        if (b) b.setAttribute('aria-expanded', 'false');
      }
    });
  }
  toggles.forEach(function (btn) {
    btn.addEventListener('click', function (ev) {
      ev.stopPropagation();
      var li = btn.closest('.has-sub');
      var open = !li.classList.contains('open');
      closeSubs(li);
      li.classList.toggle('open', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  });
  document.addEventListener('click', function () { closeSubs(null); });
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape') {
      closeSubs(null);
      if (navBtn && navBtn.getAttribute('aria-expanded') === 'true') { setNav(false); navBtn.focus(); }
    }
  });
})();

/* Copy buttons in the "How to cite" box (hidden without JS). */
(function () {
  'use strict';
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    var label = btn.textContent;
    btn.hidden = false;
    btn.addEventListener('click', function () {
      var target = document.getElementById(btn.getAttribute('data-copy'));
      if (!target) return;
      var text = target.textContent;
      var done = function () {
        btn.textContent = btn.getAttribute('data-copied') || label;
        setTimeout(function () { btn.textContent = label; }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var range = document.createRange();
        range.selectNodeContents(target);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
        sel.removeAllRanges();
      }
    });
  });
})();
