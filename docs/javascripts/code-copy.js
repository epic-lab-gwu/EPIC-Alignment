document.addEventListener("DOMContentLoaded", function () {
  var blocks = document.querySelectorAll(".rst-content pre");
  blocks.forEach(function (block) {
    if (block.querySelector(".code-copy-button")) {
      return;
    }

    var button = document.createElement("button");
    button.className = "code-copy-button";
    button.type = "button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code block");

    button.addEventListener("click", async function () {
      var text = block.innerText || "";
      try {
        await navigator.clipboard.writeText(text.replace(/\n$/, ""));
        button.textContent = "Copied";
        window.setTimeout(function () {
          button.textContent = "Copy";
        }, 1400);
      } catch (err) {
        button.textContent = "Failed";
        window.setTimeout(function () {
          button.textContent = "Copy";
        }, 1400);
      }
    });

    block.classList.add("has-copy-button");
    block.appendChild(button);
  });
});
