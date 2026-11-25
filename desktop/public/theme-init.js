try {
  var t = localStorage.getItem("babble-theme") === "light" ? "light" : "dark";
  if (t === "light") document.documentElement.dataset.theme = "light";
  document.documentElement.style.colorScheme = t;
} catch (e) {}
