
function renderPDF(pdfUrl) {
  const canvas = document.getElementById("pdf-canvas");
  const context = canvas.getContext("2d");

  // Get the parent container's width dynamically
  const container = document.getElementById("pdf-viewer-container");
  const containerHeight = container.offsetHeight;

  // Load the PDF
  pdfjsLib.getDocument(pdfUrl).promise.then(function (pdfDoc_) {
    const pdfDoc = pdfDoc_;
    const pageNumber = 1; // You can loop through pages if needed

    // Fetch the page
    pdfDoc.getPage(pageNumber).then(function (page) {
      const viewport = page.getViewport({ scale: containerHeight / page.getViewport({ scale: 1 }).height });

      // Set the canvas size to match the viewport size
      canvas.width = viewport.width;
      canvas.height = containerHeight; // Set the canvas width to the container's width

      // Render the page into the canvas
      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };
      page.render(renderContext);
    });
  });
}

function addAlgorithm() {
    console.log("algorithm button pressed");

    let newAlgorithm = document.createElement("div");
    newAlgorithm.classList.add("algorithm");

    let nameInputBox = document.createElement("div");
    let nameLabel = document.createTextNode("Name: ");
    let nameInput = document.createElement("input");
    nameInputBox.classList.add("name-input-box");
    nameInput.type = "text";

    let colorInputBox = document.createElement("div");
    let colorLabel = document.createTextNode("Color: ");
    let colorInput = document.createElement("input");
    colorInputBox.classList.add("color-input-box");
    colorInput.type = "color";
    colorInput.classList.add("color-picker");

    let chooseFileBox = document.createElement("div");
    let chooseFile = document.createElement("input");
    chooseFileBox.classList.add("choose-file-box");
    chooseFile.type = "file";
    chooseFile.accept = ".txt, .csv";

    let removeButtonBox = document.createElement("div");
    let removeButton = document.createElement("button");
    removeButtonBox.classList.add("remove-button-box");
    removeButton.innerText = "Remove";
    removeButton.classList.add("remove-button");
    removeButton.onclick = function() {
        newAlgorithm.remove();
    };

    nameInputBox.appendChild(nameLabel);
    nameInputBox.appendChild(nameInput);
    colorInputBox.appendChild(colorLabel);
    colorInputBox.appendChild(colorInput);
    chooseFileBox.appendChild(chooseFile);
    removeButtonBox.appendChild(removeButton);

    newAlgorithm.appendChild(nameInputBox);
    newAlgorithm.appendChild(colorInputBox);
    newAlgorithm.appendChild(chooseFileBox);
    newAlgorithm.appendChild(removeButtonBox);
    document.getElementById("container").appendChild(newAlgorithm);
}


// Render the PDF after the page loads
window.onload = function () {
  const pdfUrl = "static/pdf/plot.pdf";
  renderPDF(pdfUrl);
};
