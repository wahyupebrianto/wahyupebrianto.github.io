/* Blog category filter.  Progressive enhancement: without JS every post is
   listed; with JS the (initially hidden) filter buttons appear.
   The selected category is kept in the URL hash, e.g. /blog/#category=Deep%20Learning */
(function () {
  'use strict';

  var bar = document.querySelector('.filters');
  if (!bar) return;
  var buttons = Array.prototype.slice.call(bar.querySelectorAll('.chip-btn'));
  var items = Array.prototype.slice.call(document.querySelectorAll('.post-item'));
  var groups = Array.prototype.slice.call(document.querySelectorAll('.year-group'));
  var empty = document.querySelector('.filter-empty');

  function apply(category) {
    var shown = 0;
    items.forEach(function (li) {
      /* data-categories looks like "|AI|Deep Learning|" so matching is exact. */
      var match = !category || (li.getAttribute('data-categories') || '').indexOf('|' + category + '|') !== -1;
      li.hidden = !match;
      if (match) shown++;
    });
    groups.forEach(function (g) {
      g.hidden = !g.querySelector('.post-item:not([hidden])');
    });
    buttons.forEach(function (b) {
      b.setAttribute('aria-pressed', b.getAttribute('data-filter') === category ? 'true' : 'false');
    });
    if (empty) empty.hidden = shown !== 0;
  }

  function fromHash() {
    var m = /(?:^#|&)category=([^&]*)/.exec(window.location.hash);
    var value = '';
    if (m) { try { value = decodeURIComponent(m[1]); } catch (e) { value = ''; } }
    var known = buttons.some(function (b) { return b.getAttribute('data-filter') === value; });
    return known ? value : '';
  }

  buttons.forEach(function (b) {
    b.addEventListener('click', function () {
      var value = b.getAttribute('data-filter');
      apply(value);
      var hash = value ? '#category=' + encodeURIComponent(value) : window.location.pathname + window.location.search;
      if (window.history && history.replaceState) history.replaceState(null, '', hash);
    });
  });

  bar.hidden = false;
  apply(fromHash());
})();
