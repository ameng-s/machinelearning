const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const predictButton = document.getElementById("predictButton");
const result = document.getElementById("result");

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    result.innerHTML = "";
    result.className = "result";

    if (file) {
        const imageURL = URL.createObjectURL(file);

        imagePreview.src = imageURL;
        previewContainer.style.display = "block";
    } else {
        imagePreview.src = "";
        previewContainer.style.display = "none";
    }
});

uploadForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const file = imageInput.files[0];

    if (!file) {
        result.className = "result not-fresh";
        result.innerHTML = "<p>Please select an image first.</p>";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    predictButton.disabled = true;
    predictButton.textContent = "Processing...";
    result.className = "result loading";
    result.innerHTML = "<p>Analyzing image, please wait...</p>";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        let statusClass = "fresh";
        if (data.prediction === "Not Fresh") {
            statusClass = "not-fresh";
        } else if (data.prediction === "Highly Fresh") {
            statusClass = "highly-fresh";
        }

        result.className = `result ${statusClass}`;
        result.innerHTML = `
            <h3>Prediction Result</h3>
            <p>Freshness Level: <strong class="status-badge">${data.prediction}</strong></p>
            <p>Confidence: <strong>${data.confidence}%</strong></p>
        `;

    } catch (error) {
        result.className = "result not-fresh";
        result.innerHTML = `<p>Error: ${error.message}</p>`;
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Freshness";
    }
});