// The figures in these pages are <picture>s with a dark variant chosen by the system's colour
// scheme. The site can be switched to light or dark by hand, so follow the site's scheme instead.
(function () {
  function sync() {
    var dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    document.querySelectorAll("picture source[data-dark], picture source[media*='prefers-color-scheme: dark']").forEach(function (s) {
      s.setAttribute("data-dark", "");
      s.media = dark ? "all" : "not all";
    });
  }
  function start() {
    sync();
    new MutationObserver(sync).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
