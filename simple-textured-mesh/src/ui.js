export function createTextureCompareUI({
  onPredict,
  onShowOriginal,
  onShowNew,
  initialStatus = "Ready to load prediction.",
}) {
  const panel = document.createElement("aside");
  panel.className = "texture-ui";

  panel.innerHTML = `
    <div class="texture-ui__card">
      <p class="texture-ui__eyebrow">Texture Creator</p>
      <h1 class="texture-ui__title">Roughness Compare</h1>
      <p class="texture-ui__copy">
        Load the exported prediction assets, then switch between the original dataset roughness map and the new model output.
      </p>
      <div class="texture-ui__actions">
        <button class="texture-ui__button texture-ui__button--primary" data-action="predict">Predict</button>
        <button class="texture-ui__button" data-action="original" disabled>Original</button>
        <button class="texture-ui__button" data-action="new" disabled>New</button>
      </div>
      <p class="texture-ui__status" data-role="status">${initialStatus}</p>
    </div>
  `;

  document.body.appendChild(panel);

  const predictButton = panel.querySelector('[data-action="predict"]');
  const originalButton = panel.querySelector('[data-action="original"]');
  const newButton = panel.querySelector('[data-action="new"]');
  const status = panel.querySelector('[data-role="status"]');

  const setActiveButton = (active) => {
    for (const button of [originalButton, newButton]) {
      button.classList.toggle("is-active", button === active);
    }
  };

  predictButton.addEventListener("click", async () => {
    status.textContent = "Loading predicted roughness map...";
    try {
      await onPredict();
      originalButton.disabled = false;
      newButton.disabled = false;
      setActiveButton(newButton);
      status.textContent = "Predicted roughness loaded. Compare the original and new maps.";
    } catch (error) {
      status.textContent = error.message;
    }
  });

  originalButton.addEventListener("click", () => {
    onShowOriginal();
    setActiveButton(originalButton);
    status.textContent = "Showing the original roughness map.";
  });

  newButton.addEventListener("click", () => {
    onShowNew();
    setActiveButton(newButton);
    status.textContent = "Showing the predicted roughness map.";
  });

  return {
    setStatus(message) {
      status.textContent = message;
    },
  };
}
