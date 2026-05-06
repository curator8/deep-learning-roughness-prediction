export function createTextureCompareUI({
  onPredict,
  onSelectSample,
  onShowOriginal,
  onShowNew,
  samples = [],
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
      <div class="texture-ui__samples">
        <span>Test samples</span>
        <div class="texture-ui__sample-grid">
          ${samples.map((sample) => `
            <button class="texture-ui__sample" type="button" data-sample-id="${sample.id}">
              ${sample.label}
            </button>
          `).join("")}
        </div>
      </div>
      <label class="texture-ui__upload">
        <span>Albedo image</span>
        <input type="file" accept="image/png,image/jpeg,image/webp" data-action="upload">
      </label>
      <p class="texture-ui__status" data-role="status">${initialStatus}</p>
    </div>
  `;

  document.body.appendChild(panel);

  const predictButton = panel.querySelector('[data-action="predict"]');
  const originalButton = panel.querySelector('[data-action="original"]');
  const newButton = panel.querySelector('[data-action="new"]');
  const uploadInput = panel.querySelector('[data-action="upload"]');
  const sampleButtons = [...panel.querySelectorAll("[data-sample-id]")];
  const status = panel.querySelector('[data-role="status"]');

  let selectedFile = null;

  const setActiveButton = (active) => {
    for (const button of [originalButton, newButton]) {
      button.classList.toggle("is-active", button === active);
    }
  };

  predictButton.addEventListener("click", async () => {
    status.textContent = "Running backend prediction...";
    try {
      await onPredict(selectedFile);
      originalButton.disabled = false;
      newButton.disabled = false;
      setActiveButton(newButton);
      status.textContent = "Prediction loaded. Compare the original and new maps.";
    } catch (error) {
      status.textContent = error.message;
    }
  });

  uploadInput.addEventListener("change", () => {
    selectedFile = uploadInput.files[0] ?? null;
    for (const button of sampleButtons) {
      button.classList.remove("is-active");
    }
    status.textContent = selectedFile
      ? `Selected ${selectedFile.name}. Click Predict to call the backend.`
      : initialStatus;
  });

  for (const button of sampleButtons) {
    button.addEventListener("click", async () => {
      const sample = samples.find((item) => item.id === button.dataset.sampleId);
      if (!sample) {
        return;
      }

      status.textContent = `Loading ${sample.label}...`;
      try {
        selectedFile = await onSelectSample(sample);
        uploadInput.value = "";
        for (const sampleButton of sampleButtons) {
          sampleButton.classList.toggle("is-active", sampleButton === button);
        }
        originalButton.disabled = false;
        newButton.disabled = true;
        setActiveButton(originalButton);
        status.textContent = `Loaded ${sample.label}. Click Predict to call the backend.`;
      } catch (error) {
        status.textContent = error.message;
      }
    });
  }

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
