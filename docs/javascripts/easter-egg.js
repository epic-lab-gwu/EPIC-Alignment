document.addEventListener("DOMContentLoaded", function () {
  var buttons = document.querySelectorAll(".doc-easter-egg-button");

  function showMessage(host, text) {
    var message = host.querySelector(".doc-easter-egg-message");
    if (!message) {
      return;
    }
    message.textContent = text;
    message.classList.add("is-visible");
  }

  function launchConfetti(host) {
    var colors = ["#ff6b6b", "#ffd166", "#06d6a0", "#4dabf7", "#c77dff"];
    var count = 18;

    for (var i = 0; i < count; i += 1) {
      var piece = document.createElement("span");
      piece.className = "doc-confetti-piece";
      piece.textContent = Math.random() > 0.5 ? "•" : "✦";
      piece.style.color = colors[i % colors.length];
      piece.style.left = 50 + (Math.random() * 16 - 8) + "%";
      piece.style.top = "42px";
      piece.style.setProperty("--dx", (Math.random() * 120 - 60).toFixed(0) + "px");
      piece.style.setProperty("--dy", (-70 - Math.random() * 70).toFixed(0) + "px");
      piece.style.setProperty("--rot", (Math.random() * 360 - 180).toFixed(0) + "deg");
      piece.style.animationDelay = (Math.random() * 0.05).toFixed(3) + "s";
      host.appendChild(piece);

      window.setTimeout(function (node) {
        return function () {
          node.remove();
        };
      }(piece), 1400);
    }
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      var host = button.closest(".doc-easter-egg");
      if (!host) {
        return;
      }

      launchConfetti(host);
      showMessage(host, "Congratulations! You found this :)");
    });
  });
});
