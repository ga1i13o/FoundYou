(function () {
  "use strict";

  const copyButton = document.querySelector("[data-copy-target]");
  if (!copyButton) return;

  function fallbackCopy(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  copyButton.addEventListener("click", async function () {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    if (!target) return;

    const text = target.textContent;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        fallbackCopy(text);
      }
      copyButton.textContent = "Copied";
    } catch (_error) {
      copyButton.textContent = "Select text";
    }

    window.setTimeout(function () {
      copyButton.textContent = "Copy";
    }, 1800);
  });
})();
